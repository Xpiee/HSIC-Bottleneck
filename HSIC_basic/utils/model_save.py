import datetime
import os
import torch

def create_dirs(path, time_stamp):
    
    model_files = "exp_" + time_stamp
    log_dir = os.path.join(path, model_files)

    model_rp_dir = os.path.join(log_dir, "reports")
    model_report = os.path.join(model_rp_dir, "report")

    model_data = os.path.join(model_rp_dir, "model")

    print("Model files saved in: ", log_dir)
    print("Model report files saved in : ", model_rp_dir)
    print("Model saved in : ", model_data)

    dirs = [model_report, model_data]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    dirs = [model_report, model_data]
    return dirs


def load_model(filepath):
    model = torch.load(filepath)
    print("Loaded  [{}]".format(filepath))
    return model