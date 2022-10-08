import os
import pickle


class OptimzedContentLoader:
    def __init__(self, optimized_content_path):
        self.optimized_content_path = optimized_content_path
        self.file_list = sorted(os.listdir(self.optimized_content_path), key=lambda el: int(el.split("_")[0]))

    def __getitem__(self, index):
        file_name = os.path.join(self.optimized_content_path, f"{index}_save.pkl")
        with open(file_name, "rb") as infile:
            content = pickle.load(infile)
        return {
            "hand_verts": content["hand_verts_pred"],
            "obj_verts": content["obj_verts_pred"],
        }

    def __len__(self):
        return len(self.file_list)