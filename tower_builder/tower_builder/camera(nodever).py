"""
🏗️ Block Detection System - RealSense (v7)
============================================
[v7 변경사항]
- Block 클래스에 standardized_height, gripper_value 속성 추가
- 카메라 측정값 → 표준 규격(30, 40, 50mm)으로 자동 변환
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Literal


# ============================================================
# 데이터 클래스
# ============================================================

@dataclass
class Block:
    """감지된 블록 정보를 담는 데이터 클래스"""
    
    # 기본 정보
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center_2d: Tuple[int, int]       # 픽셀 좌표 (x, y)
    contour: np.ndarray = field(compare=False, repr=False)
    rotated_box: np.ndarray = field(compare=False, repr=False)
    
    # 형태 정보
    area: float = 0.0
    aspect_ratio: float = 0.0
    solidity: float = 0.0
    angle: float = 0.0  # 회전 각도 (도)
    
    # 3D 정보
    center_3d: Optional[Tuple[float, float, float]] = None  # (X, Y, Z) 미터
    depth: float = 0.0  # 미터
    
    # 실제 크기 (mm)
    real_width_mm: float = 0.0
    real_height_mm: float = 0.0
    size_class: str = "unknown"  # "small", "medium", "large"
    
    # UI용 (선택 순서)
    selection_order: int = 0
    
    # -------------------- 편의 속성 --------------------
    
    @property
    def size_mm(self) -> Tuple[float, float]:
        """실제 크기 (width, height) mm"""
        return (self.real_width_mm, self.real_height_mm)
    
    @property
    def side_length_mm(self) -> float:
        """한 변의 평균 길이 (mm)"""
        return (self.real_width_mm + self.real_height_mm) / 2
    
    @property
    def min_side_mm(self) -> float:
        """짧은 변의 길이 (mm) - 크기 판별용"""
        return min(self.real_width_mm, self.real_height_mm)
    
    @property
    def depth_cm(self) -> float:
        """깊이 (cm)"""
        return self.depth * 100
    
    @property
    def depth_mm(self) -> float:
        """깊이 (mm)"""
        return self.depth * 1000
    
    @property
    def center_3d_mm(self) -> Optional[Tuple[float, float, float]]:
        """3D 좌표 (mm 단위)"""
        if self.center_3d is None:
            return None
        x, y, z = self.center_3d
        return (x * 1000, y * 1000, z * 1000)
    
    @property
    def is_valid(self) -> bool:
        """유효한 깊이 정보가 있는지"""
        return self.depth > 0
    
    # -------------------- [NEW] 표준화된 높이 --------------------
    
    @property
    def standardized_height(self) -> float:
        """
        카메라 측정값을 표준 규격으로 변환
        - 45mm 이상 → 50mm (대형)
        - 35mm 이상 → 40mm (중형)  
        - 그 외 → 30mm (소형)
        """
        measured = self.min_side_mm
        
        if measured >= 45.0:
            return 50.0
        elif measured >= 35.0:
            return 40.0
        else:
            return 30.0
    
    @property
    def gripper_value(self) -> int:
        """
        블럭 크기에 따른 그리퍼 닫힘 값
        - 대형(50mm) → 580
        - 중형(40mm) → 650
        - 소형(30mm) → 680
        """
        h = self.standardized_height
        
        if h >= 50.0:
            return 580
        elif h >= 40.0:
            return 650
        else:
            return 680
    
    @property
    def size_name(self) -> str:
        """사람이 읽기 좋은 크기 이름"""
        h = self.standardized_height
        
        if h >= 50.0:
            return "대형(5cm)"
        elif h >= 40.0:
            return "중형(4cm)"
        else:
            return "소형(3cm)"
    
    def __str__(self) -> str:
        if self.is_valid:
            return (f"Block(center={self.center_2d}, "
                    f"depth={self.depth_cm:.1f}cm, "
                    f"size={self.side_length_mm:.1f}mm, "
                    f"std_h={self.standardized_height}mm)")
        return f"Block(center={self.center_2d}, no depth)"


# ============================================================
# 카메라 클래스
# ============================================================

class RealSenseCamera:
    """RealSense 카메라 제어 클래스"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._intrinsics: Optional[rs.intrinsics] = None
        self._depth_scale: float = 0.001
        self._is_running: bool = False
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def intrinsics(self) -> Optional[rs.intrinsics]:
        return self._intrinsics
    
    @property
    def depth_scale(self) -> float:
        return self._depth_scale
    
    def start(self) -> bool:
        """카메라 시작"""
        if self._is_running:
            return True
            
        try:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, 
                               rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, 
                               rs.format.z16, self.fps)
            
            profile = self._pipeline.start(config)
            self._align = rs.align(rs.stream.color)
            
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()
            
            depth_stream = profile.get_stream(rs.stream.depth)
            self._intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            print("📷 카메라 초기화 중...")
            for _ in range(30):
                self._pipeline.wait_for_frames()
            
            self._is_running = True
            print(f"✅ RealSense 시작! ({self.width}x{self.height})")
            print(f"   Depth Scale: {self._depth_scale:.6f}")
            return True
            
        except Exception as e:
            print(f"❌ 카메라 오류: {e}")
            return False
    
    def stop(self):
        """카메라 정지"""
        if self._pipeline and self._is_running:
            self._pipeline.stop()
            self._is_running = False
            print("📷 카메라 정지")
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """정렬된 컬러/뎁스 프레임 반환"""
        if not self._is_running:
            return None, None
            
        try:
            frames = self._pipeline.wait_for_frames()
            aligned = self._align.process(frames)
            
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception:
            return None, None
    
    def get_depth_at(self, x: int, y: int, depth_image: np.ndarray, 
                     use_sampling: bool = True) -> float:
        """특정 픽셀의 깊이값 반환 (미터)"""
        x, y = int(x), int(y)
        
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        
        raw_depth = depth_image[y, x]
        depth_m = raw_depth * self._depth_scale
        
        if depth_m > 0.05:
            return depth_m
        
        if not use_sampling:
            return 0.0
        
        offsets = [
            (-5, 0), (5, 0), (0, -5), (0, 5),
            (-10, 0), (10, 0), (0, -10), (0, 10),
            (-5, -5), (5, -5), (-5, 5), (5, 5),
            (-15, 0), (15, 0), (0, -15), (0, 15),
        ]
        
        valid_depths = []
        for dx, dy in offsets:
            sx, sy = x + dx, y + dy
            if 0 <= sx < self.width and 0 <= sy < self.height:
                raw = depth_image[sy, sx]
                d = raw * self._depth_scale
                if 0.05 < d < 3.0:
                    valid_depths.append(d)
        
        if valid_depths:
            valid_depths.sort()
            return valid_depths[len(valid_depths) // 2]
        
        return 0.0
    
    def pixel_to_3d(self, x: int, y: int, depth_image: np.ndarray
                   ) -> Optional[Tuple[float, float, float]]:
        """픽셀 좌표를 3D 좌표로 변환 (미터)"""
        depth = self.get_depth_at(x, y, depth_image)
        
        if depth <= 0 or self._intrinsics is None:
            return None
        
        point = rs.rs2_deproject_pixel_to_point(self._intrinsics, [x, y], depth)
        return (point[0], point[1], depth)
    
    def calc_real_size(self, width_px: float, height_px: float, 
                       depth: float) -> Tuple[float, float]:
        """픽셀 크기를 실제 크기(mm)로 변환"""
        if depth <= 0 or self._intrinsics is None:
            return (0.0, 0.0)
        
        real_w = (width_px * depth * 1000) / self._intrinsics.fx
        real_h = (height_px * depth * 1000) / self._intrinsics.fy
        return (real_w, real_h)


# ============================================================
# 감지기 클래스
# ============================================================

@dataclass
class DetectorConfig:
    """감지기 설정"""
    threshold: int = 200
    
    roi_x: int = 190
    roi_y: int = 140
    roi_w: int = 230
    roi_h: int = 180
    
    min_area: int = 90
    max_area: int = 4000
    
    min_aspect: float = 0.5
    max_aspect: float = 3.0
    min_solidity: float = 0.7
    
    min_depth: float = 0.1
    max_depth: float = 2.0
    
    small_thresh: float = 30
    medium_thresh: float = 50


class BlockDetector:
    """블록 감지기"""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._binary_view: Optional[np.ndarray] = None
    
    @property
    def binary_view(self) -> Optional[np.ndarray]:
        return self._binary_view
    
    @property
    def roi(self) -> Tuple[int, int, int, int]:
        c = self.config
        return (c.roi_x, c.roi_y, c.roi_w, c.roi_h)
    
    def detect(self, frame: np.ndarray, depth_image: np.ndarray,
               camera: RealSenseCamera) -> List[Block]:
        """프레임에서 블록 감지"""
        cfg = self.config
        blocks = []
        
        roi = frame[cfg.roi_y:cfg.roi_y+cfg.roi_h, 
                   cfg.roi_x:cfg.roi_x+cfg.roi_w]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, cfg.threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self._binary_view = binary
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            block = self._process_contour(cnt, depth_image, camera)
            if block is not None:
                blocks.append(block)
        
        return blocks
    
    def _process_contour(self, cnt: np.ndarray, depth_image: np.ndarray,
                         camera: RealSenseCamera) -> Optional[Block]:
        """단일 컨투어 처리"""
        cfg = self.config
        
        area = cv2.contourArea(cnt)
        if not (cfg.min_area < area < cfg.max_area):
            return None
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (_, _), (w, h), angle = rect
        
        if h == 0 or w == 0:
            return None
        
        aspect = max(w, h) / min(w, h)
        if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
            return None
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return None
        solidity = area / hull_area
        if solidity < cfg.min_solidity:
            return None
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if not (3 <= len(approx) <= 8):
            return None
        
        box_global = box.copy()
        box_global[:, 0] += cfg.roi_x
        box_global[:, 1] += cfg.roi_y
        
        cnt_global = cnt.copy()
        cnt_global[:, :, 0] += cfg.roi_x
        cnt_global[:, :, 1] += cfg.roi_y
        
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"]) + cfg.roi_x
        cy = int(M["m01"] / M["m00"]) + cfg.roi_y
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        block = Block(
            bbox=(x + cfg.roi_x, y + cfg.roi_y, bw, bh),
            center_2d=(cx, cy),
            contour=cnt_global,
            rotated_box=box_global,
            area=area,
            aspect_ratio=aspect,
            solidity=solidity,
            angle=angle
        )
        
        point_3d = camera.pixel_to_3d(cx, cy, depth_image)
        
        if point_3d:
            block.center_3d = point_3d
            block.depth = point_3d[2]
            
            if cfg.min_depth < block.depth < cfg.max_depth:
                real_w, real_h = camera.calc_real_size(w, h, block.depth)
                block.real_width_mm = real_w
                block.real_height_mm = real_h
                
                side = (real_w + real_h) / 2
                if side < cfg.small_thresh:
                    block.size_class = "small"
                elif side < cfg.medium_thresh:
                    block.size_class = "medium"
                else:
                    block.size_class = "large"
        
        return block


# ============================================================
# 통합 시스템 클래스
# ============================================================

class BlockDetectionSystem:
    """블록 감지 통합 시스템"""
    
    def __init__(self, 
                 camera_width: int = 640,
                 camera_height: int = 480,
                 camera_fps: int = 30,
                 config: Optional[DetectorConfig] = None):
        self._camera = RealSenseCamera(camera_width, camera_height, camera_fps)
        self._detector = BlockDetector(config)
        
        self._last_frame: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_blocks: List[Block] = []
    
    def __enter__(self) -> "BlockDetectionSystem":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    def start(self) -> bool:
        return self._camera.start()
    
    def stop(self):
        self._camera.stop()
        cv2.destroyAllWindows()
    
    @property
    def is_running(self) -> bool:
        return self._camera.is_running
    
    @property
    def config(self) -> DetectorConfig:
        return self._detector.config
    
    @config.setter
    def config(self, value: DetectorConfig):
        self._detector.config = value
    
    @property
    def camera(self) -> RealSenseCamera:
        return self._camera
    
    @property
    def detector(self) -> BlockDetector:
        return self._detector
    
    def update(self) -> bool:
        """새 프레임을 가져와서 블록 감지 수행"""
        color, depth = self._camera.get_frames()
        if color is None:
            return False
        
        self._last_frame = color
        self._last_depth = depth
        self._last_blocks = self._detector.detect(color, depth, self._camera)
        return True
    
    def get_blocks(self, update: bool = True) -> List[Block]:
        if update:
            self.update()
        return self._last_blocks.copy()
    
    def get_valid_blocks(self, update: bool = True) -> List[Block]:
        blocks = self.get_blocks(update)
        return [b for b in blocks if b.is_valid]
    
    def get_closest_block(self, update: bool = True) -> Optional[Block]:
        blocks = self.get_valid_blocks(update)
        if not blocks:
            return None
        return min(blocks, key=lambda b: b.depth)
    
    def get_farthest_block(self, update: bool = True) -> Optional[Block]:
        blocks = self.get_valid_blocks(update)
        if not blocks:
            return None
        return max(blocks, key=lambda b: b.depth)
    
    def get_largest_block(self, update: bool = True) -> Optional[Block]:
        blocks = self.get_blocks(update)
        if not blocks:
            return None
        return max(blocks, key=lambda b: b.area)
    
    def get_smallest_block(self, update: bool = True) -> Optional[Block]:
        blocks = self.get_blocks(update)
        if not blocks:
            return None
        return min(blocks, key=lambda b: b.area)
    
    def get_block_count(self, update: bool = True) -> int:
        return len(self.get_blocks(update))
    
    def find_blocks_by_size(self, 
                            size: Literal["small", "medium", "large"],
                            update: bool = True) -> List[Block]:
        blocks = self.get_valid_blocks(update)
        return [b for b in blocks if b.size_class == size]
    
    def find_blocks_in_depth_range(self,
                                   min_depth: float = 0,
                                   max_depth: float = float('inf'),
                                   update: bool = True) -> List[Block]:
        blocks = self.get_valid_blocks(update)
        return [b for b in blocks if min_depth <= b.depth <= max_depth]
    
    def find_block_at(self, x: int, y: int, 
                      update: bool = False) -> Optional[Block]:
        """특정 픽셀 위치의 블록 반환"""
        blocks = self.get_blocks(update)
        point = (x, y)
        
        for block in blocks:
            result = cv2.pointPolygonTest(block.rotated_box, point, False)
            if result >= 0:
                return block
        
        return None
    
    @property
    def last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame
    
    @property
    def last_depth(self) -> Optional[np.ndarray]:
        return self._last_depth
    
    @property
    def last_blocks(self) -> List[Block]:
        return self._last_blocks.copy()
    
    def get_depth_at(self, x: int, y: int) -> float:
        if self._last_depth is None:
            return 0.0
        return self._camera.get_depth_at(x, y, self._last_depth)
    
    def get_3d_at(self, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        if self._last_depth is None:
            return None
        return self._camera.pixel_to_3d(x, y, self._last_depth)


# ============================================================
# 메인 (독립 실행 시 - 디버그용)
# ============================================================

def main():
    """독립 실행 시 간단한 테스트"""
    print("\n" + "=" * 50)
    print("🏗️ Block Detection System v7 - Test")
    print("=" * 50)
    
    with BlockDetectionSystem() as system:
        print("Press ESC to exit...")
        
        while True:
            if not system.update():
                continue
            
            frame = system.last_frame.copy()
            blocks = system.last_blocks
            
            for block in blocks:
                cv2.drawContours(frame, [block.rotated_box], 0, (0, 255, 0), 2)
                cx, cy = block.center_2d
                
                if block.is_valid:
                    # [NEW] standardized_height, gripper_value 표시
                    txt = f"{block.size_name} (grip:{block.gripper_value})"
                    cv2.putText(frame, txt, (cx - 50, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Blocks: {len(blocks)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Test", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break


if __name__ == "__main__":
    main()