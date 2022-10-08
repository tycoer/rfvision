import numpy as np


def create_vertex_color(contact_info, mode="vertex_contact"):
    if mode == "vertex_contact":
        vertex_contact = contact_info["vertex_contact"]
        n_verts = vertex_contact.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[vertex_contact == 0] = np.array([57, 57, 57]) / 255.0
        vertex_color[vertex_contact == 1] = np.array([198, 198, 198]) / 255.0
        return vertex_color
    elif mode == "contact_region":
        contact_region = contact_info["hand_region"]
        n_verts = contact_region.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[contact_region == 0] = np.array([117, 0, 0]) / 255.0
        vertex_color[contact_region == 1] = np.array([255, 0, 0]) / 255.0
        vertex_color[contact_region == 2] = np.array([255, 138, 137]) / 255.0

        vertex_color[contact_region == 3] = np.array([117, 65, 0]) / 255.0
        vertex_color[contact_region == 4] = np.array([255, 144, 0]) / 255.0
        vertex_color[contact_region == 5] = np.array([255, 206, 134]) / 255.0

        vertex_color[contact_region == 6] = np.array([116, 117, 0]) / 255.0
        vertex_color[contact_region == 7] = np.array([255, 255, 0]) / 255.0
        vertex_color[contact_region == 8] = np.array([255, 255, 131]) / 255.0

        vertex_color[contact_region == 9] = np.array([0, 117, 0]) / 255.0
        vertex_color[contact_region == 10] = np.array([0, 255, 0]) / 255.0
        vertex_color[contact_region == 11] = np.array([145, 255, 133]) / 255.0

        vertex_color[contact_region == 12] = np.array([0, 60, 118]) / 255.0
        vertex_color[contact_region == 13] = np.array([0, 133, 255]) / 255.0
        vertex_color[contact_region == 14] = np.array([136, 200, 255]) / 255.0

        vertex_color[contact_region == 15] = np.array([70, 0, 118]) / 255.0
        vertex_color[contact_region == 16] = np.array([210, 135, 255]) / 255.0

        vertex_color[contact_region == 17] = np.array([255, 232, 246]) / 255.0
        return vertex_color
    else:
        raise ValueError(f"Unknown color mode: {mode}")
