import setting as st

""" model """
from model_arch import bagNet


def construct_model(config, flag_model_num = 0):
    """ construct model """
    if flag_model_num == 0:
        model_num = st.model_num_0
    elif flag_model_num == 1:
        model_num = st.model_num_1

    if model_num == 0:
        pass

    elif model_num == 6:
        model = bagNet.bagNet9(config).cuda()
    elif model_num == 7:
        model = bagNet.bagNet17(config).cuda()
    elif model_num == 8:
        model = bagNet.bagNet33(config).cuda()

    return model
