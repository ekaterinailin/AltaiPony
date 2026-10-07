#!/bin/bash
# Run a command with a memory guardrail, so that a runaway job (e.g. a GP fit
# on a long light curve) is stopped instead of freezing the machine or taking
# the editor down with it.
#
#   1. The job runs in its own systemd scope with a hard memory cap and no swap:
#      if it exceeds the cap, the kernel kills the job, and only the job.
#   2. choom marks it as the first process to kill if the whole system runs out
#      of memory, even while the job itself is under its cap.
#   3. A watchdog kills it early if the system's available memory falls below a
#      threshold, e.g. because other programs need memory too.
#
# Optionally waits until enough memory is free before starting. Matplotlib
# runs without windows (MPLBACKEND=Agg) and BLAS/OpenMP threads are capped.
#
# Requires Linux with a systemd user session (systemd-run --user) and choom
# (util-linux).
#
# Usage:
#   scripts/guarded_run.sh [options] -- command [args...]
#
# Options:
#   -m MEM      memory cap of the job, e.g. 4G (default 4G)
#   -a MB       stop the job if available system memory falls below MB
#               (default 1000)
#   -w MB       wait until at least MB are available before starting
#               (default 0, start immediately)
#   -t N        threads for OpenMP/OpenBLAS/MKL (default 4)
#   -l FILE     write the job's output to FILE instead of the terminal
#
# Examples:
#   scripts/guarded_run.sh -- python -m pytest altaipony/tests/detrending -m "not slow"
#   cd notebooks && ../scripts/guarded_run.sh -w 2000 -l injrec.log -- python -u injection_recovery.py
#
# The last line printed is "RESULT exit=<code> peak_mem=<MB>MB", preceded by
# a WATCHDOG line if the watchdog stopped the job, or a KILLED line if the job
# was killed otherwise (usually by the memory cap). The script exits with the
# job's exit code (137 if killed).

mem=4G min_avail=1000 wait_avail=0 threads=4 log=""
while getopts "m:a:w:t:l:h" opt; do
  case $opt in
    m) mem=$OPTARG ;;
    a) min_avail=$OPTARG ;;
    w) wait_avail=$OPTARG ;;
    t) threads=$OPTARG ;;
    l) log=$OPTARG ;;
    *) sed -n '2,39p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[[ $1 == "--" ]] && shift
if (( $# == 0 )); then
  echo "guarded_run.sh: no command given (see -h)" >&2; exit 2
fi
for tool in systemd-run choom; do
  command -v $tool > /dev/null || { echo "guarded_run.sh: $tool not found" >&2; exit 2; }
done

available() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }

# Messages go to the terminal and, if given, to the log file.
say() { if [[ -n $log ]]; then echo "$@" | tee -a "$log"; else echo "$@"; fi; }

if (( wait_avail > 0 )); then
  while (( $(available) < wait_avail )); do
    echo "waiting for ${wait_avail} MB available memory (now $(available) MB)"
    sleep 30
  done
fi

export MPLBACKEND=Agg OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
unit=guarded-run-$$
[[ -n $log ]] && : > "$log"
if [[ -n $log ]]; then
  systemd-run --user --scope --quiet --unit=$unit -p MemoryMax=$mem -p MemorySwapMax=0 \
    choom -n 1000 -- "$@" > "$log" 2>&1 &
else
  systemd-run --user --scope --quiet --unit=$unit -p MemoryMax=$mem -p MemorySwapMax=0 \
    choom -n 1000 -- "$@" &
fi
pid=$!

# Watchdog: stop the job if the system runs low on memory; track its peak use.
peak=0 stopped=0
while kill -0 $pid 2> /dev/null; do
  avail=$(available)
  cur=$(systemctl --user show $unit.scope -p MemoryCurrent --value 2> /dev/null)
  [[ $cur =~ ^[0-9]+$ ]] && (( cur / 1048576 > peak )) && peak=$((cur / 1048576))
  if (( avail < min_avail )); then
    say "WATCHDOG: available memory ${avail} MB < ${min_avail} MB, stopping the job"
    systemctl --user kill --signal=SIGKILL $unit.scope
    stopped=1
    break
  fi
  sleep 2
done
wait $pid
rc=$?
if (( rc == 137 && ! stopped )); then
  say "KILLED: the job was killed (SIGKILL), most likely for exceeding the ${mem} memory cap"
fi
say "RESULT exit=$rc peak_mem=${peak}MB"
exit $rc
