#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters


@dataclass
class Block:
    side_mm: float
    angle_deg: float
    depth_mm: float
    center_px: Tuple[int, int]
    center_cam_mm: Tuple[float, float, float]
    rect_w_px: float
    rect_h_px: float
    box_pts: np.ndarray
    label: str


class BlockDetectionSystem(Node):
    """
    - color + aligned depth 동기화 수신(message_filters)
    - camera_info는 별도 subscription으로 intrinsics 갱신
    - ROI만 검출
    - 블럭 클릭 시 block 콜백, 빈 곳 클릭 시 point 콜백
    """

    def __init__(self):
        super().__init__('block_detection_system')

        # =========================
        # 토픽명
        # =========================
        self.color_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        self.info_topic  = '/camera/camera/color/camera_info'

        # =========================
        # 검출 파라미터
        # =========================
        self.min_area_px = 100
        self.max_area_px = 1000

        self.roi_half = 4
        self.thresh_val = 180
        self.blur_ksize = 5

        # ROI 비율(필요시 조절)
        self.roi_x0_ratio = 0.33
        self.roi_x1_ratio = 0.70
        self.roi_y0_ratio = 0.25
        self.roi_y1_ratio = 0.68

        self.size_ranges_cm = [
            ("SMALL_3cm",  2.5, 3.5),
            ("MEDIUM_4cm", 3.5, 4.5),
            ("LARGE_5cm",  4.5, 5.5),
        ]

        # =========================
        # 내부 상태
        # =========================
        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self._last_blocks: List[Block] = []
        self._clicked_point: Optional[Tuple[int, int]] = None

        self._win_name = "realsense_color"
        self._last_color_bgr = None
        self._last_bin_img = None
        self._last_depth_m = None

        self.on_block_clicked = None   # f(Block) -> None
        self.on_point_clicked = None   # f((X,Y,Z)mm) -> None

        self._last_warn_t = 0.0

        # =========================
        # camera_info: 별도 subscription (QoS: sensor)
        # =========================
        self.sub_info = self.create_subscription(
            CameraInfo,
            self.info_topic,
            self._cb_info,
            qos_profile_sensor_data
        )

        # =========================
        # color/depth: message_filters 동기화 (QoS: sensor)
        # =========================
        self.sub_color = message_filters.Subscriber(
            self, Image, self.color_topic, qos_profile=qos_profile_sensor_data
        )
        self.sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth],
            queue_size=30,
            slop=0.20
        )
        self.ts.registerCallback(self._cb_synced)

        # =========================
        # GUI
        # =========================
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win_name, 1280, 960)
        cv2.setMouseCallback(self._win_name, self._on_mouse)

        self.get_logger().info("✅ BlockDetectionSystem started correctly")
        self.get_logger().info(f"color_topic={self.color_topic}")
        self.get_logger().info(f"depth_topic={self.depth_topic}")
        self.get_logger().info(f"info_topic ={self.info_topic}")

        self.timer = self.create_timer(0.03, self._tick)

    # -------------------------
    # camera_info callback
    # -------------------------
    def _cb_info(self, info_msg: CameraInfo):
        self.fx = float(info_msg.k[0])
        self.fy = float(info_msg.k[4])
        self.cx = float(info_msg.k[2])
        self.cy = float(info_msg.k[5])

    # -------------------------
    # synced callback (color + depth)
    # -------------------------
    def _cb_synced(self, color_msg: Image, depth_msg: Image):
        color_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        if depth_msg.encoding == '16UC1':
            depth_m = depth.astype(np.float32) / 1000.0
        elif depth_msg.encoding == '32FC1':
            depth_m = depth.astype(np.float32)
        else:
            depth_m = depth.astype(np.float32)

        self._last_color_bgr = color_bgr
        self._last_depth_m = depth_m

        # intrinsics가 아직 없으면, 화면만 띄우고 검출은 잠시 대기
        if self.fx is None:
            now = time.time()
            if now - self._last_warn_t > 1.0:
                self.get_logger().warn("camera_info(intrinsics) 수신 전입니다. 잠시 대기 중...")
                self._last_warn_t = now
            self._last_blocks = []
            self._last_bin_img = None
            return

        blocks, bin_img = self._detect_blocks_from_frames(color_bgr, depth_m)
        self._last_blocks = blocks
        self._last_bin_img = bin_img

    # -------------------------
    # Mouse
    # -------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self._clicked_point = (x, y)

        # 1) 블럭 클릭이면 블럭 콜백
        chosen = self._pick_block_at(x, y)
        if chosen is not None:
            self._print_block(chosen)
            if self.on_block_clicked is not None:
                try:
                    self.on_block_clicked(chosen)
                except Exception as e:
                    print(f"[CLICK] on_block_clicked error: {e}")
            return

        # 2) 블럭이 아니면 point 콜백(빈 곳 클릭)
        if self.on_point_clicked is None:
            return
        if self._last_depth_m is None:
            return
        if self.fx is None:
            return

        d_m = self._depth_at_median(self._last_depth_m, x, y)
        if d_m <= 0.0:
            return

        cam_xyz_mm = self._deproject_to_cam_mm(x, y, d_m)
        try:
            self.on_point_clicked(cam_xyz_mm)
        except Exception as e:
            print(f"[CLICK] on_point_clicked error: {e}")

    def _pick_block_at(self, x: int, y: int) -> Optional[Block]:
        if not self._last_blocks:
            return None
        for blk in self._last_blocks:
            inside = cv2.pointPolygonTest(
                blk.box_pts.astype(np.float32),
                (float(x), float(y)),
                False
            )
            if inside >= 0:
                return blk
        return None

    def _print_block(self, blk: Block):
        X, Y, Z = blk.center_cam_mm
        print(
            f"[CLICK] label={blk.label} | "
            f"side={blk.side_mm:.1f} mm | "
            f"angle={blk.angle_deg:.1f} deg | "
            f"center_point=({X:.1f}, {Y:.1f}) | "
            f"depth(Z)={Z:.1f} mm"
        )

    # -------------------------
    # Core math
    # -------------------------
    def _depth_at_median(self, depth_m: np.ndarray, cx: int, cy: int) -> float:
        r = self.roi_half
        h, w = depth_m.shape[:2]
        x0, x1 = max(cx - r, 0), min(cx + r + 1, w)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, h)

        roi = depth_m[y0:y1, x0:x1].astype(np.float32)
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid))

    def _px_to_m(self, px_len: float, depth_m: float) -> float:
        if depth_m <= 0.0 or self.fx is None:
            return 0.0
        return float(px_len) * depth_m / float(self.fx)

    def _deproject_to_cam_mm(self, u: int, v: int, depth_m: float) -> Tuple[float, float, float]:
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return (0.0, 0.0, 0.0)

        X = (float(u) - self.cx) * depth_m / self.fx
        Y = (float(v) - self.cy) * depth_m / self.fy
        Z = depth_m

        return (X * 1000.0, Y * 1000.0, Z * 1000.0)

    def _classify_by_cm(self, side_mm: float) -> str:
        side_cm = side_mm / 10.0
        for label, lo, hi in self.size_ranges_cm:
            if lo <= side_cm < hi:
                return label
        return "UNKNOWN"

    def _angle_to_vertical_zero(self, angle_deg_from_x: float) -> float:
        a = angle_deg_from_x - 90.0
        while a > 90.0:
            a -= 180.0
        while a <= -90.0:
            a += 180.0
        return a

    # -------------------------
    # Detection
    # -------------------------
    def _detect_blocks_from_frames(self, color_bgr: np.ndarray, depth_m: np.ndarray):
        H, W = color_bgr.shape[:2]

        rx0 = int(W * self.roi_x0_ratio)
        rx1 = int(W * self.roi_x1_ratio)
        ry0 = int(H * self.roi_y0_ratio)
        ry1 = int(H * self.roi_y1_ratio)

        rx0 = max(0, min(W - 1, rx0))
        rx1 = max(rx0 + 1, min(W, rx1))
        ry0 = max(0, min(H - 1, ry0))
        ry1 = max(ry0 + 1, min(H, ry1))

        roi_color = color_bgr[ry0:ry1, rx0:rx1]
        roi_depth = depth_m[ry0:ry1, rx0:rx1]

        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize and self.blur_ksize >= 3:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        _, bin_img = cv2.threshold(gray, self.thresh_val, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks: List[Block] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_px or area > self.max_area_px:
                continue

            rect = cv2.minAreaRect(c)  # ROI 좌표계
            (cx, cy), (rw, rh), angle = rect

            cx_i = int(round(cx)) + rx0
            cy_i = int(round(cy)) + ry0

            if rw < rh:
                angle_deg = float(angle)
            else:
                angle_deg = float(angle + 90.0)

            angle_deg = self._angle_to_vertical_zero(angle_deg)

            d_m = self._depth_at_median(roi_depth, int(round(cx)), int(round(cy)))
            if d_m <= 0.0:
                continue

            center_cam_mm = self._deproject_to_cam_mm(cx_i, cy_i, d_m)

            # side(mm) 계산( intrinsics 없으면 0 처리 ) - 지금은 fx가 있으니 정상 계산됨
            side_px = float(min(rw, rh))
            side_len_m = self._px_to_m(side_px, d_m)
            side_mm = side_len_m * 1000.0
            depth_mm = d_m * 1000.0

            box = cv2.boxPoints(rect)
            box[:, 0] += rx0
            box[:, 1] += ry0
            box = np.intp(box)

            label = self._classify_by_cm(side_mm)

            blocks.append(
                Block(
                    side_mm=float(side_mm),
                    angle_deg=float(angle_deg),
                    depth_mm=float(depth_mm),
                    center_px=(cx_i, cy_i),
                    center_cam_mm=center_cam_mm,
                    rect_w_px=float(rw),
                    rect_h_px=float(rh),
                    box_pts=box,
                    label=label,
                )
            )

        return blocks, bin_img

    # -------------------------
    # UI tick
    # -------------------------
    def _tick(self):
        if self._last_color_bgr is None:
            cv2.waitKey(1)
            return

        vis = self._last_color_bgr.copy()

        H, W = vis.shape[:2]
        rx0 = int(W * self.roi_x0_ratio)
        rx1 = int(W * self.roi_x1_ratio)
        ry0 = int(H * self.roi_y0_ratio)
        ry1 = int(H * self.roi_y1_ratio)
        cv2.rectangle(vis, (rx0, ry0), (rx1, ry1), (255, 0, 0), 2)

        for blk in self._last_blocks:
            cv2.drawContours(vis, [blk.box_pts], 0, (0, 255, 0), 2)
            cx, cy = blk.center_px
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

            txt = f"{blk.label} | {blk.side_mm:.0f}mm"
            cv2.putText(
                vis, txt,
                (max(cx - 80, 5), max(cy - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 0), 2
            )

        cv2.imshow(self._win_name, vis)
        if self._last_bin_img is not None:
            cv2.imshow("binary", self._last_bin_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = BlockDetectionSystem()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
