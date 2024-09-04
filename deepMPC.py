import do_mpc
import tensorflow as tf
import os
import tf2onnx
import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pylab as plt
from casadi import *


#--------------------------------------------------------------------------------------------#
# Convert the model to ONNX format
def get_sym_tf_model(tf_model, obs_space):
    model_input_signature = [tf.TensorSpec(np.array((1,obs_space)), name='input'),]
    output_path = os.path.join('models', 'model.onnx')
    tf_model.output_names=['output']
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, output_path=output_path, input_signature=model_input_signature)
    casadi_model = do_mpc.sysid.ONNXConversion(onnx_model)
    return casadi_model
#--------------------------------------------------------------------------------------------#
if __name__=="__main__":
    reference_joints = pd.read_csv(r"reference.csv")

    tf_model = load_model('robot_model_mpc.keras')
    # Show the DNN model architecture
    tf_model.summary()
    print(tf_model.input_shape)
    casadi_from_tf = get_sym_tf_model(tf_model, tf_model.input_shape[1])
    # Init MPC Model
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)
    X = model.set_variable(var_type='_x', var_name='X', shape=(3,1)) # X(k+1)
    x = model.set_variable(var_type='_x', var_name='x', shape=(3,1)) #X(k)
    u = model.set_variable(var_type='_u', var_name='u', shape=(3,1)) # u(k)
    q1g = model.set_variable(var_type='_tvp', var_name='q1g') #Desired Joints
    q2g = model.set_variable(var_type='_tvp', var_name='q2g')
    q3g = model.set_variable(var_type='_tvp', var_name='q3g')

    # Define RHS of the model x_K+1 = f(x_K, U_k)
    casadi_from_tf.convert(input=transpose(vertcat(x,u)))
    res = casadi_from_tf['output']
    model.set_rhs('X', res) #next state
    model.set_rhs('x', X)
    model.setup()

    # Configure the MPC
    mpc = do_mpc.controller.MPC(model)
    # Optimizer parameters#
    setup_mpc = {
        'n_horizon': 30,
        't_step': 0.01,
        'n_robust': 1,
        'store_full_solution': True,
        'state_discretization': 'discrete',

    }
    mpc.set_param(**setup_mpc)

    mpc.set_param(nlpsol_opts={'ipopt.print_level': 0, 'ipopt.tol': 1e-6, 'ipopt.max_iter': 100})

    # # Parameters for Gaussian noise
    mean = 0
    std = 0.0
    # # Adding Gaussian noise
    noisy_reference_joints = reference_joints.apply(lambda reference_joints: reference_joints + np.random.normal(mean, std, size=reference_joints.shape))
    print(noisy_reference_joints.shape)
           

    tvp_template = mpc.get_tvp_template()
    control_mode = 'trajectory' # trajectory or point_stabilization
    def tvp_fun(t_now):
            # Compute the index
            index = int(t_now // mpc.settings.t_step)
            if control_mode =='trajectory':
                    if index < (reference_joints.shape[0]-(mpc.settings.n_horizon+1)):
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q1g'] = noisy_reference_joints['q1g'].iloc[index:index+mpc.settings.n_horizon+1].tolist()
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q2g'] = noisy_reference_joints['q2g'].iloc[index:index+mpc.settings.n_horizon+1].tolist()
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q3g'] = noisy_reference_joints['q3g'].iloc[index:index+mpc.settings.n_horizon+1].tolist()

                    else:
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q1g'] = noisy_reference_joints['q1g'].iloc[-(mpc.settings.n_horizon+1):].tolist()
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q2g'] = noisy_reference_joints['q2g'].iloc[-(mpc.settings.n_horizon+1):].tolist()
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q3g'] = noisy_reference_joints['q3g'].iloc[-(mpc.settings.n_horizon+1):].tolist()
        
            if control_mode =='point_stabilization':

                    if index <= 500:
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q1g'] = -2.27e-01
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q2g'] = -1.27e-01
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q3g'] = 2.26e-01

                    elif index <=1000:
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q1g'] = -5.62e-02
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q2g'] = -4.96e-02
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q3g'] = 3.88e-01

                    elif index <=1500:
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q1g'] = 8.10e-02
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q2g'] = 8.76e-02
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q3g'] = 3.72e-01

                    else:
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q1g'] = 1.1e-01
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q2g'] = 2.55e-01
                            tvp_template['_tvp',0:mpc.settings.n_horizon+1, 'q3g'] = 1.55e-01

            
            return tvp_template
    mpc.set_tvp_fun(tvp_fun)

    Q1g = model.tvp['q1g']	
    Q2g = model.tvp['q2g']
    Q3g = model.tvp['q3g']
    mterm = 100*(X[0]-Q1g)**2 + 100*(X[1]-Q2g)**2 + 100*(X[2]-Q3g)**2 #Mayer (terminal state)
    lterm = 100*(X[0]-Q1g)**2 + 100*(X[1]-Q2g)**2 + 100*(X[2]-Q3g)**2  #Mayer (terminal state)
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-3,) #DU*R*DU R=diag(0.01, 0.01)

    # Assuming these are your six ranges for the variables

    # Lower bounds on states:
    mpc.bounds['lower','_x','X', 0] = -0.3  # Lower bound for state
    mpc.bounds['lower','_x','X', 1] = -0.3  # Lower bound for state
    mpc.bounds['lower','_x','X', 2] = 0.0  # Lower bound for state

    mpc.bounds['lower','_x','x', 0] = -0.3  # Lower bound for state
    mpc.bounds['lower','_x','x', 1] = -0.3  # Lower bound for state
    mpc.bounds['lower','_x','x', 2] = 0.0  # Lower bound for state



    # Upper bounds on states
    mpc.bounds['upper','_x','X', 0] = 0.3  # Lower bound for state
    mpc.bounds['upper','_x','X', 1] = 0.3 # Lower bound for state
    mpc.bounds['upper','_x','X', 2] = 0.3  # Lower bound for state

    mpc.bounds['upper','_x','x', 0] = 0.3  # Lower bound for state
    mpc.bounds['upper','_x','x', 1] = 0.3  # Lower bound for state
    mpc.bounds['upper','_x','x', 2] = 0.3  # Lower bound for state

    # Lower bounds on inputs:
    mpc.bounds['lower','_u', 'u', 0] = -16.0
    mpc.bounds['lower','_u', 'u', 1] = -16.0
    mpc.bounds['lower','_u', 'u', 2] = -16.0

    # Lower bounds on inputs:
    mpc.bounds['upper','_u', 'u', 0] = 16.0
    mpc.bounds['upper','_u', 'u', 1] = 16.0
    mpc.bounds['upper','_u', 'u', 2] = 16.0

    # Nonlinear Constraints
    #mpc.set_nl_cons(expr_name='x_change_limiter',expr=X[0]-1,ub=0,soft_constraint=False)

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = mpc.settings.t_step)

    # Get the template
    tvp_template2 = simulator.get_tvp_template()

    # Define the function (indexing is much simpler ...)
    def tvp_fun(t_now):
            
            return tvp_template2

    # Set the tvp_fun:
    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    # Control loop

    x0 = np.array([-0.24,-0.076,0.12, 0, 0, 0]).reshape(-1,1)
    print(x0.shape)
    simulator.x0 = x0
    mpc.x0 = x0

    mpc.set_initial_guess()

    u0 = np.zeros((3,1))
    for i in range(reference_joints.shape[0]):
        simulator.make_step(u0)

    u0 = mpc.make_step(x0)

    simulator.reset_history()
    simulator.x0 = x0
    mpc.reset_history()
    sim_time = noisy_reference_joints.shape[0]
    for i in range(sim_time):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)
        print("iteration: {}/{}".format(i, sim_time))

    states = mpc.data['_x', 'x']
    control = mpc.data['_u']

    para = mpc.data['_tvp']
    data_to_plot = np.hstack([states, control, para])
    # Convert numpy array to pandas DataFrame
    df = pd.DataFrame(data_to_plot)
    # Save DataFrame to .csv
    df.to_csv('data_to_plot.csv', index=False)



