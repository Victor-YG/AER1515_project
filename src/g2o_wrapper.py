import g2o
import numpy as np


# class Constraint():
#     def __init__(self, pose_id, point_id, u_l, v_l, u_r, v_r):
#         self.pose_id = pose_id
#         self.point_id = point_id
#         self.u_l = u_l
#         self.v_l = v_l
#         self.u_r = u_r
#         self.v_r = v_r


class StereoBundleAdjustment():
    def __init__(self):
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer = g2o.SparseOptimizer()
        self.optimizer.set_algorithm(algorithm)


    def optimize(self, max_iterations=10):
        self.optimizer.initialize_optimization()
        self.optimizer.set_verbose(True)
        self.optimizer.optimize(max_iterations)


    def define_optimization(self, cam, poses, points, constraints):
        self.add_poses(cam, poses)
        self.add_points(points)
        self.add_edges(constraints)
        print("[DBUG]: num vertices = ", len(self.optimizer.vertices()))
        print("[DUBG]: num edges    = ", len(self.optimizer.edges()))


    def add_poses(self, cam, poses):
        g2o.VertexSCam.set_cam(cam[0, 0], cam[1, 1], cam[0, 2], cam[1, 2], cam[2, 2])
        self.num_poses = len(poses)
        self.add_pose(0, poses[0], fixed=True)
        for i in range(1, len(poses)):
            self.add_pose(i, poses[i], fixed=False)


    def add_points(self, points):
        self.num_points = len(points)
        for i in range(len(points)):
            self.add_point(self.num_poses + i, points[i])


    def add_edges(self, constraints):
        for i in range(len(constraints)):
            cst = constraints[i]
            self.add_edge(
                point_id=cst["point_id"],
                pose_id=cst["pose_id"],
                measurement=np.array([cst["u_l"], cst["v_l"]]))


    def add_pose(self, pose_id, pose, fixed=False):
        R = pose[0:3, 0:3]
        t = pose[0:3, 3].reshape([3, 1])
        pose_SE3 = g2o.Isometry3d(R, t)
        v_se3 = g2o.VertexSCam()
        v_se3.set_id(pose_id)
        v_se3.set_estimate(pose_SE3)
        v_se3.set_fixed(fixed)
        v_se3.set_all()
        self.optimizer.add_vertex(v_se3)

        # R = pose[0:3, 0:3]
        # t = pose[0:3, 3].reshape([3, 1])
        # pose_SE3 = g2o.Isometry3d(R, t)
        # sbacam = g2o.SBACam(pose_SE3.orientation(), pose_SE3.position())
        # sbacam.set_cam(cam[0, 0], cam[1, 1], cam[0, 2], cam[1, 2], cam[2, 2]) #TODO::assumed [2, 2] is the baseline
        # v_se3 = g2o.VertexCam()
        # v_se3.set_id(pose_id * 2) # internal id
        # v_se3.set_estimate(sbacam)
        # v_se3.set_fixed(fixed)
        # self.optimizer.add_vertex(v_se3)


    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id)
        v_p.set_marginalized(marginalized)
        v_p.set_estimate(point)
        self.optimizer.add_vertex(v_p)

        # v_p = g2o.VertexSBAPointXYZ()
        # v_p.set_id(point_id * 2 + 1)
        # v_p.set_estimate(point)
        # v_p.set_marginalized(marginalized)
        # v_p.set_fixed(fixed)
        # self.optimizer.add_vertex(v_p)


    def add_edge(self, point_id, pose_id, measurement, information=np.identity(2), robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))): # 95% CI
        edge = g2o.EdgeProjectP2MC()
        # edge = g2o.EdgeProjectP2SC() # TODO::switch to use stereo camera
        edge.set_vertex(0, self.optimizer.vertex(point_id))
        edge.set_vertex(1, self.optimizer.vertex(pose_id))
        edge.set_measurement(measurement) # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        self.optimizer.add_edge(edge)

        # edge = g2o.EdgeProjectP2MC()
        # # edge = g2o.EdgeProjectP2SC() # TODO::switch to use stereo camera
        # edge.set_vertex(0, self.optimizer.vertex(point_id * 2 + 1))
        # edge.set_vertex(1, self.optimizer.vertex(pose_id * 2))
        # edge.set_measurement(measurement) # projection
        # edge.set_information(information)

        # if robust_kernel is not None:
        #     edge.set_robust_kernel(robust_kernel)
        # self.optimizer.add_edge(edge)


    def get_poses(self):
        poses = []

        for i in range(self.num_poses):
            pose = self.get_pose(i)
            poses.append(pose)

        return poses


    def get_points(self):
        points = []

        for i in range(self.num_points):
            point = self.get_point(i)
            points.append(point)

        return points


    def get_pose(self, pose_id):
        return self.optimizer.vertex(pose_id).estimate()
        # return self.optimizer.vertex(pose_id * 2).estimate()


    def get_point(self, point_id):
        return self.optimizer.vertex(self.num_poses + point_id).estimate()
        # return self.optimizer.vertex(point_id * 2 + 1).estimate()
