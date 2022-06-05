import rowan

def get_imagenet_mean_std():
    """Gives mean and std of images from the Image Net dataset"""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return mean, std

def get_translation_from_imu(initial_velocity, cur_accel, timestep):
    """ Calculate displacement using kinematics equation s = ut + 1/2 at^2"""
    return initial_velocity * timestep + 1/2 * cur_accel * timestep**2

def integrate_angular_velocity(intial_orientation, angular_vel, timestep):
    """Integrate angular velocity based on previous quartenion rotation"""
    return rowan.calculus.integrate(intial_orientation, angular_vel, timestep)