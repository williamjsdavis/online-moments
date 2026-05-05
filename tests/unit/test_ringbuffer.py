import numpy as np

from online_moments.ringbuffer import push, push_int


def test_push_basic():
    buf = np.zeros(4)
    push(buf, 1.0)
    assert list(buf) == [1.0, 0.0, 0.0, 0.0]
    push(buf, 2.0)
    assert list(buf) == [2.0, 1.0, 0.0, 0.0]
    push(buf, 3.0)
    push(buf, 4.0)
    assert list(buf) == [4.0, 3.0, 2.0, 1.0]


def test_push_overflow_drops_oldest():
    buf = np.zeros(3)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        push(buf, v)
    assert list(buf) == [5.0, 4.0, 3.0]


def test_push_int():
    buf = np.zeros(3, dtype=np.int64)
    for v in [10, 20, 30, 40]:
        push_int(buf, v)
    assert list(buf) == [40, 30, 20]


def test_push_empty_buf_no_error():
    buf = np.zeros(0)
    push(buf, 1.0)  # no error


def test_buf_length_one():
    buf = np.zeros(1)
    push(buf, 7.0)
    assert buf[0] == 7.0
    push(buf, 9.0)
    assert buf[0] == 9.0
