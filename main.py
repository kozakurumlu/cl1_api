"""
Simple CL1 API simulation example.

Opens a CL connection, sends a biphasic stimulation pulse to a few channels,
runs a short closed loop, and prints spikes detected per channel.
"""

import cl

STIM_CHANNELS = [10, 20, 30]
LOOP_HZ = 100
LOOP_SECONDS = 1


def main():
    with cl.open() as neurons:
        print(f"CL1 connected  (sample rate: {neurons.get_frames_per_second()} Hz)")
        print(f"Channels: {neurons.get_channel_count()}\n")

        # -- Run closed loop, stimulate on first tick, then count spikes --
        targets = cl.ChannelSet(STIM_CHANNELS)
        pulse = cl.StimDesign(180, -1.5, 180, 1.5)
        spike_counts: dict[int, int] = {}
        total_spikes = 0

        for tick in neurons.loop(LOOP_HZ, stop_after_seconds=LOOP_SECONDS):
            if tick.iteration == 0:
                neurons.stim(targets, pulse)
                print(f"Sent biphasic pulse to channels {STIM_CHANNELS}\n")

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

        print("\nDone.")


if __name__ == "__main__":
    main()
