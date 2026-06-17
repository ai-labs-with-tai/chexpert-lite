MULTILABEL_DISEASES = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

DISEASE_VN_MAP = {
    "Atelectasis": "Xẹp phổi (Atelectasis)",
    "Cardiomegaly": "Tim to (Cardiomegaly)",
    "Consolidation": "Đông đặc phổi (Consolidation)",
    "Edema": "Phù phổi (Edema)",
    "Pleural_Effusion": "Tràn dịch màng phổi (Pleural Effusion)",
    "Pleural Effusion": "Tràn dịch màng phổi (Pleural Effusion)",
}


def get_vn_name(english_name):
    return DISEASE_VN_MAP.get(english_name, english_name)
