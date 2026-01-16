# import json
# import math

# INPUT = "data/raw/metadata/shorts_metadata.json"
# OUTPUT = "data/processed/dataset.json"

# with open(INPUT) as f:
#     data = json.load(f)

# dataset = []

# for item in data:
#     views = item["views"]
#     likes = item["likes"]
#     comments = item["comments"]

#     engagement = (
#         0.4 * math.log(views + 1)
#         + 0.4 * (likes / (views + 1))
#         + 0.2 * (comments / (views + 1))
#     )

#     dataset.append({
#         "video_id": item["video_id"],
#         "engagement": engagement
#     })

# with open(OUTPUT, "w") as f:
#     json.dump(dataset, f, indent=2)

# print("Dataset built successfully")
