import pickle
import os


def load_example_hist(path):
    if os.path.exists(path):
        path = f"{path}data"
        if os.path.exists(path):
            # Unpickling
            with open(path, "rb") as fp:
                example_hist = pickle.load(fp)
            return example_hist
    #os.makedirs(path)
    #print("created directory for saving experience later..")
    return []
    

def save_example_hist(dir_path, example_hist):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    path = f"{dir_path}data"
    # Pickling
    with open(path, "wb") as fp:
        pickle.dump(example_hist, fp)


def add_new_examples(dir_path, new_examples):
    example_hist = load_example_hist(dir_path)
    if example_hist:
        if len(example_hist[-1]) < (len(example_hist[-2]) if len(example_hist)>2 else 200):
            example_hist[-1] += new_examples
        else:
            example_hist.append(new_examples)
    else:
        example_hist = [new_examples]
    save_example_hist(dir_path, example_hist)
