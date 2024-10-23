import json
import os
from argparser import get_args
args = get_args()
results = {}
class_names = ['__background__', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']
for f in os.listdir(args.input_path):
    if f.endswith("_json"):
        with open(os.path.join(args.input_path, f)) as fp:
            epoch = int(f.split("-")[1].split("_")[0])
            data = json.load(fp)
            results[int(epoch)] = {
                "precision" : {
                    class_names[1] : data["1"][0],
                    class_names[2] : data["2"][0],
                    class_names[3] : data["3"][0]
                },
                "recall" : {
                    class_names[1] : data["1"][1],
                    class_names[2] : data["2"][1],
                    class_names[3] : data["3"][1]
                },
                "ap50" : {
                    class_names[1] : data["1"][2],
                    class_names[2] : data["2"][2],
                    class_names[3] : data["3"][2]
                }
            }
sorted_dict = dict(sorted(results.items()))
final_results = {"metrics": list(sorted_dict.values())}
with open(args.output_path + "/results.json", "w") as fp:
    json.dump(final_results, fp)