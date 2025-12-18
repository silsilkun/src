import cv2
import numpy as np
import pyrealsense2 as rs
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
from contextlib import contextmanager
from datetime import datetime

# ===== Global System Handle =====
_system = None

# 데이터 클래스

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
    
    # 메타 정보
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    click_order: int = 0  # 클릭 순서
    
    # -------------------- 편의 속성 --------------------
    
    @property
    def side_length_mm(self) -> float:
        """한 변의 평균 길이 (mm)"""
        return (self.real_width_mm + self.real_height_mm) / 2
    
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
        
        
        
    def copy_with_click_order(self, order: int) -> "Block":
        """
        클릭 순서를 포함한 안전한 Block 복사
        (GUI / 프레임 변경과 분리됨)
        """
        data = asdict(self)

        # numpy array는 asdict로 deepcopy 안 되므로 그대로 유지
        data["contour"] = self.contour
        data["rotated_box"] = self.rotated_box

        data["click_order"] = order
        return Block(**data)
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환 (저장용)"""
        return {
            'click_order': self.click_order,
            'timestamp': self.timestamp,
            'center_2d_x': self.center_2d[0],
            'center_2d_y': self.center_2d[1],
            'bbox_x': self.bbox[0],
            'bbox_y': self.bbox[1],
            'bbox_w': self.bbox[2],
            'bbox_h': self.bbox[3],
            'area': self.area,
            'aspect_ratio': self.aspect_ratio,
            'solidity': self.solidity,
            'angle': self.angle,
            'depth_m': self.depth,
            'depth_cm': self.depth_cm,
            'depth_mm': self.depth_mm,
            'center_3d_x_mm': self.center_3d_mm[0] if self.center_3d_mm else None,
            'center_3d_y_mm': self.center_3d_mm[1] if self.center_3d_mm else None,
            'center_3d_z_mm': self.center_3d_mm[2] if self.center_3d_mm else None,
            'real_width_mm': self.real_width_mm,
            'real_height_mm': self.real_height_mm,
            'side_length_mm': self.side_length_mm,
        }
    
    def __str__(self) -> str:
        if self.is_valid:
            return (f"Block(order={self.click_order}, center={self.center_2d}, "
                    f"depth={self.depth_cm:.1f}cm, "
                    f"size={self.side_length_mm:.1f}mm)")
        return f"Block(order={self.click_order}, center={self.center_2d}, no depth)"

# 바닥 클릭 데이터 클래스

@dataclass
class FloorPoint:
    """바닥(빈 공간) 클릭 정보를 저장"""

    pixel: Tuple[int, int]                      # (x, y)
    depth: float                                # meters
    point_3d: Optional[Tuple[float, float, float]]  # (X, Y, Z) meters
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def point_3d_mm(self):
        if self.point_3d is None:
            return None
        x, y, z = self.point_3d
        return (x * 1000, y * 1000, z * 1000)


# 카메라 클래스

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
    
    # -------------------- 속성 --------------------
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def intrinsics(self) -> Optional[rs.intrinsics]:
        return self._intrinsics
    
    @property
    def depth_scale(self) -> float:
        return self._depth_scale
    
    # -------------------- 생명주기 --------------------
    
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
            
            # Depth scale 가져오기
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()
            
            # Intrinsics 가져오기
            depth_stream = profile.get_stream(rs.stream.depth)
            self._intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            # 워밍업
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
    
    # -------------------- 프레임 획득 --------------------
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        정렬된 컬러/뎁스 프레임 반환
        
        Returns:
            (color_image, depth_image) 또는 (None, None)
        """
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
    
    # -------------------- 깊이 측정 --------------------
    
    def get_depth_at(self, x: int, y: int, depth_image: np.ndarray, 
                     use_sampling: bool = True) -> float:
        """
        특정 픽셀의 깊이값 반환 (미터)
        
        Args:
            x, y: 픽셀 좌표
            depth_image: 깊이 이미지 (uint16)
            use_sampling: 중심값이 0일 때 주변 샘플링 사용 여부
        """
        x, y = int(x), int(y)
        
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        
        # 중심점 값
        raw_depth = depth_image[y, x]
        depth_m = raw_depth * self._depth_scale
        
        if depth_m > 0.05:  # 5cm 이상이면 유효
            return depth_m
        
        if not use_sampling:
            return 0.0
        
        # 주변 샘플링
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
            return valid_depths[len(valid_depths) // 2]  # 중앙값
        
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


# 감지기 클래스

@dataclass
class DetectorConfig:
    """감지기 설정"""
    # 이진화
    threshold: int = 200
    
    # ROI (Region of Interest)
    roi_x: int = 190
    roi_y: int = 140
    roi_w: int = 230
    roi_h: int = 180
    
    # 면적 필터
    min_area: int = 90
    max_area: int = 4000
    
    # 형태 필터
    min_aspect: float = 0.5
    max_aspect: float = 3.0
    min_solidity: float = 0.7
    
    # 깊이 필터
    min_depth: float = 0.1
    max_depth: float = 2.0


class BlockDetector:
    """블록 감지기"""
    


    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._binary_view: Optional[np.ndarray] = None
    
    @property
    def binary_view(self) -> Optional[np.ndarray]:
        """최근 이진화 이미지 (디버깅용)"""
        return self._binary_view
    
    @property
    def roi(self) -> Tuple[int, int, int, int]:
        """현재 ROI (x, y, w, h)"""
        c = self.config
        return (c.roi_x, c.roi_y, c.roi_w, c.roi_h)
    
    def detect(self, frame: np.ndarray, depth_image: np.ndarray,
               camera: RealSenseCamera) -> List[Block]:
        """
        프레임에서 블록 감지
        
        Args:
            frame: BGR 컬러 이미지
            depth_image: 깊이 이미지 (uint16)
            camera: RealSenseCamera 인스턴스
            
        Returns:
            감지된 Block 리스트
        """
        cfg = self.config
        blocks = []
        
        # ROI 추출
        roi = frame[cfg.roi_y:cfg.roi_y+cfg.roi_h, 
                   cfg.roi_x:cfg.roi_x+cfg.roi_w]
        
        # 전처리
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, cfg.threshold, 255, cv2.THRESH_BINARY)
        
        # 모폴로지 연산
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self._binary_view = binary
        
        # 컨투어 찾기
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
        
        # 면적 필터
        area = cv2.contourArea(cnt)
        if not (cfg.min_area < area < cfg.max_area):
            return None
        
        # 회전 사각형
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (_, _), (w, h), angle = rect
        
        if h == 0 or w == 0:
            return None
        
        # 종횡비 필터
        aspect = max(w, h) / min(w, h)
        if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
            return None
        
        # Solidity 필터
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return None
        solidity = area / hull_area
        if solidity < cfg.min_solidity:
            return None
        
        # 꼭지점 수 필터
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if not (3 <= len(approx) <= 8):
            return None
        
        # 전역 좌표로 변환
        box_global = box.copy()
        box_global[:, 0] += cfg.roi_x
        box_global[:, 1] += cfg.roi_y
        
        cnt_global = cnt.copy()
        cnt_global[:, :, 0] += cfg.roi_x
        cnt_global[:, :, 1] += cfg.roi_y
        
        # 중심점 계산
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"]) + cfg.roi_x
        cy = int(M["m01"] / M["m00"]) + cfg.roi_y
        
        # 바운딩 박스
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Block 생성
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
        
        # 3D 정보 추가
        point_3d = camera.pixel_to_3d(cx, cy, depth_image)
        
        if point_3d:
            block.center_3d = point_3d
            block.depth = point_3d[2]
            
            if cfg.min_depth < block.depth < cfg.max_depth:
                # minAreaRect의 w, h 사용
                real_w, real_h = camera.calc_real_size(w, h, block.depth)
                block.real_width_mm = real_w
                block.real_height_mm = real_h
        
        return block
    

# 통합 시스템 클래스

class BlockDetectionSystem:
    
    def __init__(self, 
                 camera_width: int = 640,
                 camera_height: int = 480,
                 camera_fps: int = 30,
                 config: Optional[DetectorConfig] = None):

        self._camera = RealSenseCamera(camera_width, camera_height, camera_fps)
        self._detector = BlockDetector(config)
        
        # 캐시
        self._last_frame: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_blocks: List[Block] = []
        
        # 클릭한 블록 저장 리스트
        self._clicked_blocks: List[Block] = []
        self._clicked_floor_points: List[FloorPoint] = []
    
    # -------------------- Context Manager --------------------
    
    def __enter__(self) -> "BlockDetectionSystem":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    # -------------------- 생명주기 --------------------
    
    def start(self) -> bool:
        """시스템 시작"""
        return self._camera.start()
    
    def stop(self):
        """시스템 정지"""
        self._camera.stop()
        cv2.destroyAllWindows()
    
    @property
    def is_running(self) -> bool:
        return self._camera.is_running
    
    # -------------------- 설정 접근 --------------------
    
    @property
    def config(self) -> DetectorConfig:
        """감지기 설정"""
        return self._detector.config
    
    @config.setter
    def config(self, value: DetectorConfig):
        self._detector.config = value
    
    @property
    def camera(self) -> RealSenseCamera:
        """카메라 인스턴스 (고급 사용)"""
        return self._camera
    
    @property
    def detector(self) -> BlockDetector:
        """감지기 인스턴스 (고급 사용)"""
        return self._detector
    
    # -------------------- 핵심 기능 --------------------
    
    def update(self) -> bool:
        """
        새 프레임을 가져와서 블록 감지 수행
        
        Returns:
            성공 여부
        """
        color, depth = self._camera.get_frames()
        if color is None:
            return False
        
        self._last_frame = color
        self._last_depth = depth
        self._last_blocks = self._detector.detect(color, depth, self._camera)
        return True
    
    def get_blocks(self, update: bool = True) -> List[Block]:
        """
        감지된 블록 리스트 반환
        
        Args:
            update: True면 새 프레임으로 갱신 후 반환
        """
        if update:
            self.update()
        return self._last_blocks.copy()
    
    def get_valid_blocks(self, update: bool = True) -> List[Block]:
        """유효한 깊이 정보가 있는 블록만 반환"""
        blocks = self.get_blocks(update)
        return [b for b in blocks if b.is_valid]
    
    def _is_already_clicked(self, block: Block) -> bool:
        """이미 클릭된 블록인지 확인 (bbox 기준)"""
        for b in self._clicked_blocks:
            if b.bbox == block.bbox:
                return True
        return False
    # -------------------- 클릭 블록 관리 --------------------
    
    def get_clicked_blocks(self) -> List[Block]:
        """클릭한 블록 리스트 반환"""
        return self._clicked_blocks.copy()
    
    def get_clicked_floor_points(self) -> List[FloorPoint]:
        """클릭한 바닥 포인트 리스트 반환"""
        return self._clicked_floor_points.copy()
    
    def clear_clicked_blocks(self):
        """클릭한 블록 리스트 초기화"""
        self._clicked_blocks.clear()
        self._clicked_floor_points.clear()
        print("🗑️  클릭 블록 리스트 초기화됨")
    
   
    def print_clicked_blocks_summary(self):
        """클릭한 블록들의 요약 정보 출력"""
        if not self._clicked_blocks:
            print("⚠️  클릭한 블록이 없습니다")
            return
        
        print("\n" + "=" * 60)
        print(f"📊 클릭한 블록 요약 (총 {len(self._clicked_blocks)}개)")
        print("=" * 60)
        
        for i, block in enumerate(self._clicked_blocks, 1):
            print(f"\n[{i}] {block}")
            if block.is_valid and block.center_3d_mm:
                x, y, z = block.center_3d_mm
                print(f"    위치: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
                print(f"    크기: {block.side_length_mm:.1f} mm")
                print(f"    각도: {block.angle:.1f} °")
        
        print("=" * 60 + "\n")

    def print_clicked_floor_points_summary(self):
        """클릭한 바닥 포인트 요약 출력"""
        if not self._clicked_floor_points:
            print("⚠️  클릭한 바닥 포인트가 없습니다")
            return

        print("\n" + "=" * 60)
        print(f"🟦 클릭한 바닥 포인트 요약 (총 {len(self._clicked_floor_points)}개)")
        print("=" * 60)

        for i, fp in enumerate(self._clicked_floor_points, 1):
            print(f"\n[{i}] 픽셀 좌표: {fp.pixel}")

            if fp.point_3d_mm:
                x, y, z = fp.point_3d_mm
                print(f"    3D 좌표: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
                print(f"    깊이: {fp.depth * 100:.1f} cm")
            else:
                print("    ⚠️ 깊이 정보 없음")

        print("=" * 60)
    
    # -------------------- 편의 메서드 --------------------
    
    def get_closest_block(self, update: bool = True) -> Optional[Block]:
        """가장 가까운 블록 반환"""
        blocks = self.get_valid_blocks(update)
        if not blocks:
            return None
        return min(blocks, key=lambda b: b.depth)
    
    def get_farthest_block(self, update: bool = True) -> Optional[Block]:
        """가장 먼 블록 반환"""
        blocks = self.get_valid_blocks(update)
        if not blocks:
            return None
        return max(blocks, key=lambda b: b.depth)
    
    def get_largest_block(self, update: bool = True) -> Optional[Block]:
        """가장 큰 블록 반환 (면적 기준)"""
        blocks = self.get_blocks(update)
        if not blocks:
            return None
        return max(blocks, key=lambda b: b.area)
    
    def get_smallest_block(self, update: bool = True) -> Optional[Block]:
        """가장 작은 블록 반환 (면적 기준)"""
        blocks = self.get_blocks(update)
        if not blocks:
            return None
        return min(blocks, key=lambda b: b.area)
    
    def get_block_count(self, update: bool = True) -> int:
        """감지된 블록 수"""
        return len(self.get_blocks(update))
    
    # -------------------- 필터링 --------------------
    
    def find_blocks_in_depth_range(self,
                                   min_depth: float = 0,
                                   max_depth: float = float('inf'),
                                   update: bool = True) -> List[Block]:
        """특정 깊이 범위의 블록들 반환 (미터 단위)"""
        blocks = self.get_valid_blocks(update)
        return [b for b in blocks if min_depth <= b.depth <= max_depth]
    
    def find_block_at(self, x: int, y: int, 
                      update: bool = False) -> Optional[Block]:
        """특정 픽셀 위치의 블록 반환"""
        blocks = self.get_blocks(update)
        for block in blocks:
            bx, by, bw, bh = block.bbox
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return block
        return None
    
    # -------------------- 프레임 접근 --------------------
    
    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """마지막 컬러 프레임"""
        return self._last_frame
    
    @property
    def last_depth(self) -> Optional[np.ndarray]:
        """마지막 깊이 프레임"""
        return self._last_depth
    
    @property
    def last_blocks(self) -> List[Block]:
        """마지막 감지 결과"""
        return self._last_blocks.copy()
    
    # -------------------- 깊이 조회 --------------------
    
    def get_depth_at(self, x: int, y: int) -> float:
        """특정 픽셀의 깊이값 (미터)"""
        if self._last_depth is None:
            return 0.0
        return self._camera.get_depth_at(x, y, self._last_depth)
    
    def get_3d_at(self, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        """특정 픽셀의 3D 좌표 (미터)"""
        if self._last_depth is None:
            return None
        return self._camera.pixel_to_3d(x, y, self._last_depth)
    
    # -------------------- 디버그 GUI --------------------
    
    def run_debug(self):
        """
        디버그 GUI 실행
        
        조작:
            - 블록 클릭: 상세 정보 출력 및 리스트 저장
            - 빈 공간 클릭: 해당 위치 깊이 출력
            - 트랙바: 감지 파라미터 조정
            - 'p' 키: 클릭한 블록 요약 출력
            - 'c' 키: 클릭한 블록 리스트 초기화
            - ESC: 종료
        """
        print("\n" + "=" * 50)
        print("🏗️ Block Detection - Debug Mode")
        print("=" * 50)
        print("📌 조작:")
        print("   - 블록 클릭: 상세 정보 + 리스트 저장")
        print("   - 빈 공간 클릭: 깊이 확인")
        print("   - 'p' 키: 저장된 블록 요약")
        print("   - 'c' 키: 저장 리스트 초기화")
        print("   - ESC: 종료")
        print("=" * 50 + "\n")
        
        selected_idx = -1
        
        def on_mouse(event, x, y, flags, param):
            nonlocal selected_idx

            if event != cv2.EVENT_LBUTTONDOWN:
                return

            # =========================
            # 1️⃣ 블록 클릭 검사
            # =========================
            for i, block in enumerate(self._last_blocks):
                bx, by, bw, bh = block.bbox
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    selected_idx = i

                    if self._is_already_clicked(block):
                        print("⚠️ 이미 클릭된 블록입니다")
                        return

                    block_copy = block.copy_with_click_order(
                        len(self._clicked_blocks) + 1
                    )

                    self._clicked_blocks.append(block_copy)
                    self._print_block_info(block_copy)

                    print(f"💾 블록 저장 완료 (총 {len(self._clicked_blocks)}개)")
                    return  # ⭐ 블록이면 여기서 반드시 종료

            # =========================
            # 2️⃣ 빈 공간 클릭 (바닥)
            # =========================
            selected_idx = -1

            depth = self.get_depth_at(x, y)
            point_3d = self.get_3d_at(x, y)

            floor_point = FloorPoint(
                pixel=(x, y),
                depth=depth,
                point_3d=point_3d
            )

            self._clicked_floor_points.append(floor_point)

            print("\n🟦 바닥 클릭 저장")
            print(f"  픽셀: ({x}, {y})")

            if point_3d:
                X, Y, Z = floor_point.point_3d_mm
                print(f"  3D 좌표: X={X:.1f}mm Y={Y:.1f}mm Z={Z:.1f}mm")
            else:
                print("  깊이 없음")

            print(f"  총 바닥 클릭 수: {len(self._clicked_floor_points)}")


                    
                    
        
        cv2.namedWindow("Result")
        cv2.setMouseCallback("Result", on_mouse)
        
        cv2.namedWindow("Control")
        cv2.createTrackbar("Threshold", "Control", 
                          self.config.threshold, 255, lambda x: None)
        cv2.createTrackbar("Min Area", "Control", 
                          self.config.min_area, 5000, lambda x: None)
        cv2.createTrackbar("Max Area", "Control", 
                          self.config.max_area, 30000, lambda x: None)
        
        try:
            while True:
                # 트랙바 값 적용
                self.config.threshold = cv2.getTrackbarPos("Threshold", "Control")
                self.config.min_area = cv2.getTrackbarPos("Min Area", "Control")
                self.config.max_area = cv2.getTrackbarPos("Max Area", "Control")
                
                # 업데이트
                if not self.update():
                    continue
                
                # 결과 그리기
                display = self._draw_result(selected_idx)
                
                # 클릭 카운트 표시
                cv2.putText(display, f"Clicked: {len(self._clicked_blocks)}", 
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                cv2.imshow("Result", display)
                
                # 이진화 이미지
                if self._detector.binary_view is not None:
                    cv2.imshow("Binary (ROI)", self._detector.binary_view)
                
                # 깊이 시각화
                depth_display = self._draw_depth()
                cv2.imshow("Depth", depth_display)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('p'):  # 요약 출력
                    self.print_clicked_blocks_summary()
                    self.print_clicked_floor_points_summary()
                elif key == ord('c'):  # 초기화
                    self.clear_clicked_blocks()
                    
        except KeyboardInterrupt:
            pass
        finally:
            if self._clicked_blocks:
                print(f"\n💾 {len(self._clicked_blocks)}개 블록이 저장되어 있습니다")
            
            cv2.destroyAllWindows()
            print("👋 디버그 모드 종료")
    
    def _draw_result(self, selected_idx: int = -1) -> np.ndarray:
        """결과 이미지 그리기"""
        display = self._last_frame.copy()
        cfg = self.config
        
        # ROI 표시
        cv2.rectangle(display, 
                     (cfg.roi_x, cfg.roi_y),
                     (cfg.roi_x + cfg.roi_w, cfg.roi_y + cfg.roi_h),
                     (0, 0, 255), 2)
        cv2.putText(display, "ROI (WHITE)", (cfg.roi_x, cfg.roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 블록 그리기
        for i, block in enumerate(self._last_blocks):
            is_selected = (i == selected_idx)
            color = (0, 255, 255) if is_selected else (0, 255, 0)
            thickness = 3 if is_selected else 2
            
            cv2.drawContours(display, [block.rotated_box], 0, color, thickness)
            cx, cy = block.center_2d
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            
            if block.is_valid:
                txt_depth = f"{block.depth_cm:.0f}cm"
                cv2.putText(display, txt_depth, (cx - 15, cy - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                txt_size = f"{block.side_length_mm:.0f}mm"
                cv2.putText(display, txt_size, (cx - 20, cy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                cv2.putText(display, "no depth", (cx - 25, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            cv2.putText(display, f"({cx},{cy})", (cx - 25, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 블록 수 표시
        cv2.putText(display, f"Blocks: {len(self._last_blocks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display
    
    def _draw_depth(self) -> np.ndarray:
        """깊이 이미지 시각화"""
        depth_display = cv2.applyColorMap(
            cv2.convertScaleAbs(self._last_depth, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        valid_mask = self._last_depth > 0
        if np.any(valid_mask):
            min_d = np.min(self._last_depth[valid_mask]) * self._camera.depth_scale * 100
            max_d = np.max(self._last_depth[valid_mask]) * self._camera.depth_scale * 100
            cv2.putText(depth_display, f"Range: {min_d:.0f}-{max_d:.0f}cm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return depth_display
    
    def _print_block_info(self, block: Block):
        """블록 정보 출력"""
        print("\n" + "=" * 50)
        print(f"🎯 블록 정보 #{block.click_order}")
        print("=" * 50)
        print(f"  📍 중심점 (픽셀): {block.center_2d}")
        
        if block.is_valid:
            pos = block.center_3d_mm
            print(f"  📍 3D 좌표: X={pos[0]:.1f}mm, Y={pos[1]:.1f}mm, Z={pos[2]:.1f}mm")
            print(f"  📏 한 변 길이: {block.side_length_mm:.1f} mm")
            print(f"  📐 각도: {block.angle:.1f}°")
            print(f"  📊 면적: {block.area:.1f} px²")
        else:
            print("  ⚠️ 깊이 정보 없음")
        
        print("=" * 50 + "\n")
    
    def _print_depth_info(self, x: int, y: int):
        """빈 공간 깊이 정보 출력"""
        print(f"\n빈 공간 클릭 ({x}, {y})")
        
        depth = self.get_depth_at(x, y)
        if depth > 0:
            print(f"  → 깊이: {depth * 100:.1f}cm")
            
            point_3d = self.get_3d_at(x, y)
            if point_3d:
                X, Y, Z = point_3d
                print(f"  → 3D 좌표: X={X*1000:.1f}mm, Y={Y*1000:.1f}mm, Z={Z*1000:.1f}mm")
        else:
            print("  → 깊이 측정 불가\n")


def stop_system():
    """카메라 및 시스템 종료"""
    global _system

    if _system is not None:
        _system.stop()
        _system = None
def get_clicked_blocks():
    """저장된 블록 리스트 반환"""
    if _system is None:
        return []
    return _system.get_clicked_blocks()


def get_clicked_floor_points():
    """저장된 바닥 포인트 리스트 반환"""
    if _system is None:
        return []
    return _system.get_clicked_floor_points()

def get_block_summaries():
    """
    클릭된 블록에서 필요한 값만 추려서 리스트로 반환
    """
    if _system is None:
        raise RuntimeError("System not started")

    summaries = []

    for b in _system.get_clicked_blocks():  # 🔥 직접 접근 말고 메서드 사용
        if b.center_3d is None:
            continue

        summaries.append({
            "center_3d": tuple(float(x) for x in b.center_3d),
            "angle": float(b.angle),
            "real_width_mm": float(b.real_width_mm),
            "click_order": int(b.click_order)
        })

    return summaries

def get_floor_summaries():
    global _system
    if _system is None:
        raise RuntimeError("System not started")

    summaries = []
    for f in _system._clicked_floor_points:
        summaries.append({
            "pixel": tuple(f.pixel),
            "depth": float(f.depth),
            "point_3d": tuple(float(x) for x in f.point_3d) if f.point_3d else None,
            "timestamp": f.timestamp
        })
    return summaries


def run_gui():
    """
    디버그 GUI 실행
    (start_system()이 먼저 호출되어야 함)
    """
    if _system is None:
        raise RuntimeError("System not started. Call start_system() first.")

    _system.run_debug()


def start_system():
    """
    BlockDetectionSystem을 생성하고 카메라를 시작
    (이미 있으면 기존 인스턴스 반환)
    """
    global _system

    if _system is None:
        _system = BlockDetectionSystem()
        _system.start()

    return _system