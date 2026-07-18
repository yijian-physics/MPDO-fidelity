#!/bin/bash
# Submit all p=0.5  F X noise  jobs at once.
# $P1 = N, $P2 = floor.  (p=0.5 is set inside multilayer_optimizarion.py)
# Time limits are the recorded p=0.3 F-X-noise runtimes, rounded up with buffer.
# Edit the list if you want, then run:  bash submit_all.sh
#
#        N  floor  TIME(HH:MM:SS)   # recorded p=0.3 runtime
runs=(
  "6  1  01:00:00"   # 0:45:34
  "8  1  01:30:00"   # 1:11:11
  "10 1  02:00:00"   # 1:34:45
  "12 1  03:15:00"   # 2:37:31
  "14 1  04:30:00"   # 3:36:48
  "16 1  06:00:00"   # 5:01:45
  "18 1  08:00:00"   # 6:26:04
  "20 1  09:00:00"   # 7:03:25
  "22 1  10:30:00"   # 8:30:19
  "24 1  13:00:00"   # 10:45:31
  "26 1  16:00:00"   # 13:28:19
  "28 1  17:00:00"   # 13:44:51
  "30 1  19:00:00"   # 15:46:21
  "32 1  21:00:00"   # 17:35:23
  "6  2  02:00:00"   # 1:37:31
  "8  2  04:30:00"   # 3:29:37
  "10 2  08:00:00"   # 6:21:24
  "12 2  15:00:00"   # 12:17:01
  "14 2  28:00:00"   # 23:30:07
  "16 2  44:00:00"   # 1-12:40:58 (36h41m)
  "18 2  60:00:00"   # ~48h
  "20 2  72:00:00"   # ~60h
  "22 2  72:00:00"   # ~60h
  "24 2  72:00:00"   # ~60h
  "26 2  72:00:00"   # ~60h
  "28 2  72:00:00"   # ~60h
  "30 2  72:00:00"   # ~60h
  "32 2  72:00:00"   # ~60h
)

for r in "${runs[@]}"; do
  read p1 p2 t <<< "$r"
  echo "Submitting N=$p1 floor=$p2 time=$t"
  sbatch --time="$t" --job-name="fid_p05_N${p1}_f${p2}" --export=ALL,P1="$p1",P2="$p2" fidelity.slurm
done
