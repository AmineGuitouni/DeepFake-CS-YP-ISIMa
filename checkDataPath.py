import os

if __name__ == "__main__":
    path = "paths/paths/paths_train.txt"

    Data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            parts = l.rstrip().split(' ')
            input_image_path = parts[0].replace("[ROOT]/",'').replace("//","/")
            mask_image_path = parts[1].replace("[ROOT]/",'').replace("//","/")
            mask_edge_path = parts[2].replace("[ROOT]/",'').replace("//","/")
            label = int(parts[3])

            if os.path.exists(input_image_path) and (os.path.exists(mask_image_path)):
                Data.append(f"{input_image_path} {mask_image_path} None {label}")

            if len(Data) >= 500:
                break

    print(len(Data))
    with open("paths/train_seg_min_500.txt", "w") as f:
        for row in Data:
            f.write("".join(row) + "\n")