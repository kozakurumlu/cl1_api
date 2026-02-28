"""
Simple CL1 API simulation example.

Opens the mock CL connection, sends a biphasic stimulation pulse to a few
channels, runs a short closed loop, and prints spikes detected per channel.
"""

import mock_cl as cl

STIM_CHANNELS = [10, 20, 30]
STIM_CURRENT_UA = 800
LOOP_HZ = 100
LOOP_SECONDS = 1

with cl.open(seed=42) as neurons:
    print(f"CL1 mock connected  (sample rate: {neurons.get_frames_per_second()} Hz)")
    print(f"Active channels: {len(cl.ACTIVE_CHANNELS)}\n")

    # -- Stimulate a few channels with a biphasic pulse --
    targets = cl.ChannelSet(STIM_CHANNELS)
    pulse = cl.StimDesign(180, -STIM_CURRENT_UA, 180, STIM_CURRENT_UA)
    neurons.stim(targets, pulse)
    print(f"Sent biphasic pulse ({STIM_CURRENT_UA} uA) to channels {STIM_CHANNELS}\n")

    # -- Run closed loop and count spikes --
    spike_counts: dict[int, int] = {}
    total_spikes = 0

    for tick in neurons.loop(LOOP_HZ, stop_after_seconds=LOOP_SECONDS):
        for spike in tick.analysis.spikes:
            spike_counts[spike.channel] = spike_counts.get(spike.channel, 0) + 1
            total_spikes += 1

    # -- Print results --
    print(f"Ran {LOOP_SECONDS}s loop at {LOOP_HZ} ticks/sec")
    print(f"Total spikes detected: {total_spikes}")
    print(f"Channels that spiked: {len(spike_counts)}\n")

    if spike_counts:
        print("Top 10 most active channels:")
        for ch, count in sorted(spike_counts.items(), key=lambda x: -x[1])[:10]:
            bar = "#" * count
            print(f"  ch {ch:2d}: {count:3d} spikes  {bar}")

    print("\nDone — mock CL1 simulation is working.")
