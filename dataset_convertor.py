import glob
import os

folder_path = "E:/privacy_info/Geolife Trajectories 1.3/Data"
output_trace = "E:/privacy_info/PrivTreeTrace/simple_example.dat"
max_data_per_people = 100

with open(output_trace, "w") as output:
    for person_id in range(182):
        output.write(f"#{person_id}:\n>0:")
        write_out_data = 0
        person_folder = os.path.join(folder_path, str(person_id).zfill(3), "Trajectory")
        for file_path in glob.glob(os.path.join(person_folder, "*"), recursive=True):
            if not file_path.endswith(".plt"):
                continue
            with open(file_path, "r") as person_input:
                for _ in range(6):
                    person_input.readline()
                for row in person_input:
                    splited_row = row.split(",")
                    output.write(f"{splited_row[0]},{splited_row[1]};")
                    write_out_data = write_out_data + 1
                    if write_out_data > max_data_per_people:
                        break
                if write_out_data > max_data_per_people:
                    break
        output.write("\n")
                
    