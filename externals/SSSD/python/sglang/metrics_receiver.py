"""This is a tool to receive metrics from the SRT server."""

import threading
from queue import Empty, Queue

import zmq

from sglang.srt.server_args import PortArgs
from sglang.srt.utils.network import get_zmq_socket


class MetricsReceiver:
    def __init__(self, port_args: PortArgs):
        self.context = zmq.Context()
        self.socket = get_zmq_socket(
            self.context, zmq.PULL, port_args.metrics_ipc_name, True
        )

        self.metrics_queue = Queue()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.socket.close()
        self.context.term()

    def _receive_loop(self):
        while self.running:
            try:
                # Non-blocking receive with timeout
                if self.socket.poll(100):  # 100ms timeout
                    metrics = self.socket.recv_pyobj(zmq.NOBLOCK)
                    self.metrics_queue.put(metrics)
            except zmq.Again:
                continue
            except Exception as e:
                print(f"Error receiving metrics: {e}")
                break

    def get_latest_metrics(self):
        """Get the most recent metrics, if available"""
        latest = None
        try:
            while True:
                latest = self.metrics_queue.get_nowait()
        except Empty:
            pass
        return latest

    def get_all_metrics(self):
        """Get all accumulated metrics"""
        metrics = []
        try:
            while True:
                metrics.append(self.metrics_queue.get_nowait())
        except Empty:
            pass
        return metrics
