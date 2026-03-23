import os
import moviepy.video.io.ImageSequenceClip
import re
#image_folder='../images/NEWexplosion_u'
image_folder = "PRESI_CONDITIONED"
def extract_values(filename):
    match = re.search(r'time_step_(\d+)_comparison_([-+]?[0-9]*\.?[0-9]+)_([-+]?[0-9]*\.?[0-9]+)\.png', filename)
    if match:
        time_value = int(match.group(1))
        sx_value = float(match.group(2))
        sy_value = float(match.group(3))
        return time_value, sx_value, sy_value
    else:
        return None


image_files = [os.path.join(image_folder, img)
               for img in os.listdir(image_folder)
               if img.endswith(".png")]
# Function to extract TIME, SX, and SY from the filename


# Create a list of tuples for sorting
sorted_list = []
for i in image_files:
    values = extract_values(i)
    if values:
        sorted_list.append((values[1], values[2], values[0], i))  # (SX, SY, TIME, filename)

# Sort the list by SX, SY, and then TIME
sorted_list.sort()

# Extract the sorted filenames
sorted_image_files= [filename for sx, sy, time, filename in sorted_list]

print(sorted_image_files)
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(sorted_image_files, fps=20)
clip.write_videofile('{}/conditioned_1.mp4'.format(image_folder))
