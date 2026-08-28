import asyncio

import pytest

from narration_barrier import NarrationBarrier


@pytest.mark.asyncio
async def test_barrier_waits_for_matching_ack() -> None:
    barrier = NarrationBarrier()
    token = barrier.arm("intro_step@0")
    barrier.open_ack()
    barrier.ack("intro_step@0", token)
    assert await barrier.wait("intro_step@0", token, timeout=1.0)


@pytest.mark.asyncio
async def test_barrier_rejects_ack_before_speech_starts() -> None:
    barrier = NarrationBarrier()
    token = barrier.arm("intro_step@0")
    assert not barrier.ack("intro_step@0", token)

    async def delayed_open_and_ack() -> None:
        await asyncio.sleep(0.05)
        barrier.open_ack()
        barrier.ack("intro_step@0", token)

    asyncio.create_task(delayed_open_and_ack())
    assert await barrier.wait("intro_step@0", token, timeout=1.0)


@pytest.mark.asyncio
async def test_barrier_ignores_stale_ack() -> None:
    barrier = NarrationBarrier()
    token = barrier.arm("intro_step@0")
    barrier.open_ack()
    assert not barrier.ack("intro_step@0", token - 1)
    assert not barrier.ack("intro_step@1", token)

    async def delayed_ack() -> None:
        await asyncio.sleep(0.05)
        barrier.ack("intro_step@0", token)

    asyncio.create_task(delayed_ack())
    assert await barrier.wait("intro_step@0", token, timeout=1.0)
