import rclpy
import DR_init
import sys
from tower_builder.vtestDataS import vtestData


def main(args=None):
    rclpy.init(args=args)

    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    VEL = 60    
    ACC = 60

    node = rclpy.create_node('example_py', namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    from DSR_ROBOT2 import(
        movej, movel,
        set_robot_mode, ROBOT_MODE_AUTONOMOUS
    ) 

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    block, floor = vtestData()

    rx, ry, rz = 90, 180, 90

    # 대기 위치로 이동
    movej([0, 0, 90, 0, 90, 0], VEL, ACC)
    # 쌓기
    for i in range(len(block)):
        camx, camy, camz = block[i]["center_3d"]

        final_x = 690 + camy
        final_y = camx
        final_z = 823 - camz

        movel([final_x, final_y, final_z, rx, ry, rz], VEL, ACC)

    rclpy.shutdown()

if __name__ == '__main__':
    main()