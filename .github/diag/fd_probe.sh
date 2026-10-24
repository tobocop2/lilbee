#!/usr/bin/env bash
# Run a command under the flock-held close probe. Branch diag/fd-closer only.
# Usage: fd_probe.sh OUT_DIR COMMAND [ARG...]
# Writes OUT_DIR/bpftrace.txt and points the test probe at OUT_DIR. Exits with the command's rc,
# or 90 when bpftrace does not attach.
set -u
SUDO="${SUDO-sudo}"
out="$1"
shift
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
script="$out/fd_closer.bt"
cp "$here/fd_closer.bt" "$script"
# dup2 is a syscall on x86_64 but not on arm64; probe it only where the kernel has it.
if $SUDO bpftrace -l 'tracepoint:syscalls:sys_enter_dup2' | grep -q dup2; then
	cat >>"$script" <<'BT'

tracepoint:syscalls:sys_enter_dup2
{
	$fd = (int64)args->newfd;
	if (@held[pid, $fd]) {
		printf("FDB-HELDCLOSE pid=%d tid=%d comm=%s call=dup2 fd=%d owner_tid=%d armed=%d ns=%lu\n",
		       pid, tid, comm, $fd, @held[pid, $fd], @armed[pid], nsecs);
		printf("FDB-NATIVE-BEGIN%sFDB-NATIVE-END\n", ustack(perf, 12));
		delete(@held[pid, $fd]);
		@held_count[pid]--;
		if (@armed[pid]) {
			signal("SIGUSR2");
		}
	}
}
BT
fi
# stdout, not -o: bpftrace 0.20 does not flush an -o file until it exits.
$SUDO bpftrace --unsafe -B line "$script" >"$out/bpftrace.txt" 2>"$out/bpftrace.stderr" &
tracer=$!
waited=0
until grep -q '^FDB-READY' "$out/bpftrace.txt" 2>/dev/null; do
	if [ "$waited" -ge 120 ] || ! kill -0 "$tracer" 2>/dev/null; then
		echo "FDC-PROBE bpftrace did not attach"
		tail -20 "$out/bpftrace.stderr"
		exit 90
	fi
	sleep 1
	waited=$((waited + 1))
done
echo "FDC-PROBE attached after ${waited}s"
start=$(date +%s.%N)
LILBEE_TEST_FD_PROBE_DIR="$out" "$@"
rc=$?
end=$(date +%s.%N)
echo "FDC-TIME probed seconds=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.2f", b - a }')"
$SUDO pkill -INT -x bpftrace
for _ in $(seq 1 60); do
	grep -q '^FDB-END' "$out/bpftrace.txt" && break
	sleep 1
done
grep -q '^FDB-END' "$out/bpftrace.txt" || echo "FDC-PROBE bpftrace did not stop cleanly"
exit "$rc"
