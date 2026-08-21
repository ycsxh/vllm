# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from queue import Empty

import msgspec
import pytest

from vllm.distributed.kv_events import (
    BlockStored,
    EventBatch,
    EventPublisherFactory,
    KVEventBatch,
    NullEventPublisher,
    ZmqEventPublisher,
    ZmqEventSubscriber,
)

DP_RANK = 0


class EventSample(
    msgspec.Struct,
    tag=True,  # type: ignore
    array_like=True,  # type: ignore
):
    """Test event for publisher testing"""

    id: int
    value: str


class SampleBatch(EventBatch):
    """Test event batch for publisher testing"""

    events: list[EventSample]


def create_test_events(count: int) -> SampleBatch:
    """Create a batch of test events"""
    events = [EventSample(id=i, value=f"test-{i}") for i in range(count)]
    return SampleBatch(ts=time.time(), events=events)


def test_basic_publishing(publisher, subscriber):
    """Test basic event publishing works"""

    test_batch = create_test_events(5)
    publisher.publish(test_batch)

    result = subscriber.receive_one(timeout=1000)
    assert result is not None, "No message received"

    seq, received = result
    assert seq == 0, "Sequence number mismatch"
    assert received.ts == pytest.approx(test_batch.ts, abs=0.1), "Timestamp mismatch"
    assert len(received.events) == len(test_batch.events), "Number of events mismatch"

    for i, event in enumerate(received.events):
        assert event.id == i, "Event id mismatch"
        assert event.value == f"test-{i}", "Event value mismatch"


def test_multiple_events(publisher, subscriber):
    """Test publishing and receiving multiple event batches"""
    for _ in range(10):
        batch = create_test_events(2)
        publisher.publish(batch)

    received = []
    for _ in range(10):
        data = subscriber.receive_one(timeout=100)
        if data:
            received.append(data)

    assert len(received) == 10, "Number of messages mismatch"
    seqs = [seq for seq, _ in received]
    assert seqs == list(range(10)), "Sequence numbers mismatch"


def test_publish_drops_saturated_batch_without_waiting(monkeypatch):
    """A stalled publisher cannot backpressure the Scheduler-facing call."""
    release_consumer = threading.Event()

    def stalled_publisher_thread(self):
        release_consumer.wait()
        while self._running or not self._event_queue.empty():
            try:
                queued = self._event_queue.get(timeout=0.01)
            except Empty:
                continue
            if queued is None:
                return
            self._event_queue.task_done()

    monkeypatch.setattr(
        ZmqEventPublisher,
        "_publisher_thread",
        stalled_publisher_thread,
    )
    publisher = ZmqEventPublisher(
        data_parallel_rank=DP_RANK,
        endpoint="inproc://test-saturated-publisher",
        max_queue_size=1,
    )
    publish_returned = threading.Event()

    def publish_second_batch():
        publisher.publish(create_test_events(1))
        publish_returned.set()

    publisher.publish(create_test_events(1))
    publish_thread = threading.Thread(target=publish_second_batch)
    publish_thread.start()

    try:
        assert publish_returned.wait(timeout=0.1)
        assert publisher.stats.dropped_batches == 1
    finally:
        if not publish_returned.is_set():
            publisher._event_queue.get_nowait()
        publish_thread.join(timeout=1)
        release_consumer.set()
        publisher.shutdown()


def test_dropped_batch_creates_a_source_sequence_gap(monkeypatch):
    """A later delivered batch exposes the sequence reserved by a drop."""
    release_consumer = threading.Event()
    publisher_thread = ZmqEventPublisher._publisher_thread

    def stalled_publisher_thread(self):
        release_consumer.wait()
        publisher_thread(self)

    monkeypatch.setattr(
        ZmqEventPublisher,
        "_publisher_thread",
        stalled_publisher_thread,
    )
    endpoint = "inproc://test-publisher-sequence-gap"
    publisher = ZmqEventPublisher(
        data_parallel_rank=DP_RANK,
        endpoint=endpoint,
        max_queue_size=1,
    )
    from .conftest import MockSubscriber

    subscriber = MockSubscriber(endpoint, None, "")

    try:
        publisher.publish(create_test_events(1))
        publisher.publish(create_test_events(1))
        release_consumer.set()

        first = subscriber.receive_one(timeout=1000)
        assert first is not None
        publisher.publish(create_test_events(1))
        after_gap = subscriber.receive_one(timeout=1000)
        assert after_gap is not None
        assert (first[0], after_gap[0]) == (0, 2)
    finally:
        release_consumer.set()
        publisher.shutdown()
        subscriber.close()


def test_publisher_stats_report_queue_and_background_activity(
    publisher,
    subscriber,
):
    batch = create_test_events(3)
    publisher.publish(batch)

    assert subscriber.receive_one(timeout=1000) is not None
    deadline = time.monotonic() + 1
    while publisher.stats.published_batches == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    stats = publisher.stats
    assert stats.enqueued_batches == 1
    assert stats.enqueued_events == 3
    assert stats.enqueued_blocks == 0
    assert stats.published_batches == 1
    assert stats.published_events == 3
    assert stats.published_blocks == 0
    assert stats.published_bytes > 0
    assert stats.queue_depth == 0
    assert stats.queue_high_watermark == 1
    publisher.observe_event_construction_time(0.25)
    assert publisher.stats.total_event_construction_time_seconds == 0.25
    assert stats.total_enqueue_time_seconds >= 0
    assert stats.total_publish_lag_seconds >= 0


def test_native_subscriber_decodes_publisher_batch_without_proxy_schema():
    endpoint = "inproc://test-native-event-subscriber"
    publisher = ZmqEventPublisher(
        data_parallel_rank=DP_RANK,
        endpoint=endpoint,
        topic="tda",
    )
    subscriber = ZmqEventSubscriber(endpoint, topic="tda")
    batch = KVEventBatch(
        ts=time.time(),
        events=[
            BlockStored(
                block_hashes=[123],
                parent_block_hash=None,
                token_ids=[1, 2, 3, 4],
                block_size=4,
                lora_id=None,
                medium="GPU",
                lora_name=None,
                group_idx=0,
                kv_cache_spec_kind="full_attention",
            )
        ],
    )

    try:
        publisher.publish(batch)
        received = subscriber.receive_one(timeout=1000)
    finally:
        publisher.shutdown()
        subscriber.close()

    assert received is not None
    sequence, decoded = received
    assert sequence == 0
    assert decoded == batch
    assert publisher.stats.enqueued_blocks == 1
    assert publisher.stats.published_blocks == 1
    assert subscriber.stats.received_batches == 1
    assert subscriber.stats.received_blocks == 1
    assert subscriber.stats.received_bytes > 0
    assert isinstance(decoded.events[0], BlockStored)


def test_native_subscriber_rejects_multiple_publishers_without_source_identity():
    with pytest.raises(ValueError, match="exactly one publisher"):
        ZmqEventSubscriber(["inproc://publisher-a", "inproc://publisher-b"])


def test_replay_mechanism(publisher, subscriber):
    """Test the replay mechanism works correctly"""
    for _ in range(19):
        batch = create_test_events(1)
        publisher.publish(batch)

    # Drain live events to ensure publisher has buffered them.
    for _ in range(19):
        assert subscriber.receive_one(timeout=1000) is not None

    subscriber.request_replay(10)

    replayed = subscriber.receive_replay()

    assert len(replayed) == 9, (
        f"Expected 9 replayed messages (seq 10-18), got {len(replayed)}"
    )
    seqs = [seq for seq, _ in replayed]
    assert seqs == list(range(10, 19)), "Replayed sequences should be 10-18"


def test_replay_includes_topic(publisher, subscriber, publisher_config):
    """Test that replay responses include the topic, matching PUB format"""
    for _ in range(5):
        publisher.publish(create_test_events(1))

    # Drain live events to ensure publisher has processed them.
    for _ in range(5):
        assert subscriber.receive_one(timeout=1000) is not None

    subscriber.request_replay(0)

    # receive_replay unpacks (topic, seq, payload) and asserts
    # topic == publisher topic for each message.
    replayed = subscriber.receive_replay()
    assert len(replayed) == 5, f"Expected 5 replayed messages, got {len(replayed)}"
    seqs = [seq for seq, _ in replayed]
    assert seqs == list(range(5)), "Replayed sequences should be 0-4"


def test_buffer_limit(publisher, subscriber, publisher_config):
    """Test buffer limit behavior"""
    buffer_size = publisher_config.buffer_steps

    # Publish more events than the buffer can hold
    for i in range(buffer_size + 10):
        batch = create_test_events(1)
        publisher.publish(batch)

    time.sleep(0.5)  # Need publisher to process above requests
    subscriber.request_replay(0)

    replayed = subscriber.receive_replay()

    assert len(replayed) == buffer_size, (
        f"Expected {buffer_size} replayed messages, got {len(replayed)}"
    )

    seqs = [seq for seq, _ in replayed]
    assert seqs == list(range(10, buffer_size + 10)), (
        "Should replay seq 11 through buffer_size+10"
    )


def test_topic_filtering(publisher_config):
    """
    Test that a subscriber only receives messages matching its topic filter
    """
    publisher_config.replay_endpoint = None

    publisher_config.topic = "foo"
    pub = EventPublisherFactory.create(publisher_config, DP_RANK)

    from .conftest import MockSubscriber

    sub_foo = MockSubscriber(publisher_config.endpoint, None, "foo")
    sub_bar = MockSubscriber(publisher_config.endpoint, None, "bar")

    try:
        time.sleep(0.1)

        for _ in range(3):
            pub.publish(create_test_events(1))

        foo_received = [sub_foo.receive_one(timeout=200) for _ in range(3)]
        assert all(msg is not None for msg in foo_received), (
            "Subscriber with matching topic should receive messages"
        )

        bar_received = [sub_bar.receive_one(timeout=200) for _ in range(3)]
        assert all(msg is None for msg in bar_received), (
            "Subscriber with non-matching topic should receive no messages"
        )
    finally:
        pub.shutdown()
        sub_foo.close()
        sub_bar.close()


def test_high_volume(publisher, subscriber):
    """Test publishing and receiving a high volume of events"""
    num_batches = 10_000
    events_per_batch = 100

    # Publish events in a separate thread to not block
    def publish_events():
        for i in range(num_batches):
            batch = create_test_events(events_per_batch)
            publisher.publish(batch)
            # Small delay to avoid overwhelming
            if i % 100 == 0:
                time.sleep(0.01)

    received: list[tuple[int, SampleBatch]] = []

    publisher_thread = threading.Thread(target=publish_events)
    publisher_thread.start()

    start_time = time.time()
    while len(received) < num_batches:
        if time.time() - start_time > 10:  # Timeout after 10 seconds
            break

        result = subscriber.receive_one(timeout=100)
        if result:
            received.append(result)

    publisher_thread.join()

    assert len(received) >= num_batches * 0.9, "We should have received most messages"

    seqs = [seq for seq, _ in received]
    assert sorted(seqs) == seqs, "Sequence numbers should be in order"


def test_null_publisher():
    """Test that NullEventPublisher can be used without errors"""
    publisher = NullEventPublisher(DP_RANK)

    # This should not raise any errors
    batch = create_test_events(5)
    publisher.publish(batch)
    publisher.shutdown()


def test_data_parallel_rank_tagging(publisher_config):
    """Test that events are properly tagged with their data parallel rank"""

    publisher_config.topic = "foo"
    pub_0 = EventPublisherFactory.create(publisher_config, DP_RANK)
    pub_1 = EventPublisherFactory.create(publisher_config, DP_RANK + 1)

    # Hardcode the expected endpoints based on port offsetting behavior
    # Both ranks get offsets according to _offset_endpoint_port function
    base_endpoint = publisher_config.endpoint
    if "tcp://" in base_endpoint:
        # For TCP endpoints: tcp://localhost:5557 -> tcp://localhost:5557, tcp://localhost:5558
        expected_endpoint_0 = base_endpoint  # rank 0 gets port + 0 = same port
        expected_endpoint_1 = base_endpoint.replace(
            ":5557", ":5558"
        )  # rank 1 gets port + 1
    else:
        # For inproc endpoints: inproc://test -> inproc://test_dp0, inproc://test_dp1
        expected_endpoint_0 = base_endpoint  # rank 0 gets base
        expected_endpoint_1 = base_endpoint + "_dp1"  # rank 1 gets _dp1

    from .conftest import MockSubscriber

    sub_0 = MockSubscriber(expected_endpoint_0, None, publisher_config.topic)
    sub_1 = MockSubscriber(expected_endpoint_1, None, publisher_config.topic)

    try:
        time.sleep(0.1)  # Let publishers start up

        # Publish events from different ranks
        batch_0 = create_test_events(2)
        batch_1 = create_test_events(3)

        pub_0.publish(batch_0)
        pub_1.publish(batch_1)

        # Receive events from rank 0
        result_0 = sub_0.receive_one(timeout=200)
        assert result_0 is not None, "No message received from rank 0"
        seq_0, received_0 = result_0

        # Receive events from rank 1
        result_1 = sub_1.receive_one(timeout=200)
        assert result_1 is not None, "No message received from rank 1"
        seq_1, received_1 = result_1

        # Verify DP rank tagging
        assert received_0.data_parallel_rank == 0, (
            f"Expected DP rank 0, got {received_0.data_parallel_rank}"
        )
        assert received_1.data_parallel_rank == 1, (
            f"Expected DP rank 1, got {received_1.data_parallel_rank}"
        )

        # Verify event content is correct
        assert len(received_0.events) == 2, "Wrong number of events from rank 0"
        assert len(received_1.events) == 3, "Wrong number of events from rank 1"

    finally:
        pub_0.shutdown()
        pub_1.shutdown()
        sub_0.close()
        sub_1.close()


def test_event_publisher_factory():
    """Test event publisher factory creation behavior under different configurations"""
    from vllm.config.kv_events import KVEventsConfig
    from vllm.distributed.kv_events import ZmqEventPublisher

    # test config is None
    publisher = EventPublisherFactory.create(None, DP_RANK)
    assert isinstance(publisher, NullEventPublisher)
    publisher.shutdown()

    # test disable kv cache events
    config = KVEventsConfig(
        enable_kv_cache_events=False,
        publisher="zmq",  # Even if zmq is specified, should return NullEventPublisher
        endpoint="tcp://localhost:5557",
    )
    publisher = EventPublisherFactory.create(config, DP_RANK)
    assert isinstance(publisher, NullEventPublisher)
    publisher.shutdown()

    # test zmq publisher
    config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="inproc://test-factory-true",
    )
    publisher = EventPublisherFactory.create(config, DP_RANK)
    assert isinstance(publisher, ZmqEventPublisher)
    publisher.shutdown()

    # test unknown publisher
    with pytest.raises(ValueError, match="Input should be"):
        KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="unknown_publisher",
            endpoint="tcp://localhost:5557",
        )

    # test publisher not specified
    config = KVEventsConfig(
        enable_kv_cache_events=True,
        # publisher not specified, should default to "zmq"
        endpoint="tcp://localhost:5557",
    )
    publisher = EventPublisherFactory.create(config, DP_RANK)
    assert isinstance(publisher, ZmqEventPublisher)
    publisher.shutdown()
