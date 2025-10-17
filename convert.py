import os
import glob
import subprocess
import shlex

# This script converts all MP4 video files in the current directory
# to MP3 audio files using ffmpeg with Nvidia hardware acceleration!

def convert_videos_to_audio(source_directory='.', output_directory='audio'):
    """
    Converts MP4 files to MP3 using ffmpeg with Nvidia hardware acceleration.

    Args:
        source_directory (str): The directory containing the video files.
        output_directory (str): The directory to save the audio files.
    """
    print("Starting video-to-audio conversion...")
    
    # Create the output directory if it doesn't exist.
    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"Ensured output directory '{output_directory}' exists.")
    except OSError as e:
        print(f"Error creating directory {output_directory}: {e}")
        return

    # Find all MP4 files in the source directory.
    # We use glob for a cleaner, cross-platform way to find files.
    mp4_files = glob.glob(os.path.join(source_directory, '*.mp4'))

    if not mp4_files:
        print(f"No MP4 files found in the directory '{source_directory}'.")
        return

    print(f"Found {len(mp4_files)} MP4 files to process.")

    for video_file in mp4_files:
        # Get the base filename without the path or extension.
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_file = os.path.join(output_directory, f"{base_name}.mp3")

        # The core ffmpeg command.
        # -hwaccel cuda: Specifies CUDA for hardware acceleration.
        # -c:v h264_cuvid: Uses the Nvidia video decoder.
        # -i: Specifies the input file.
        # -vn: Disables video processing (we only want audio).
        # -acodec libmp3lame: Sets the audio codec to MP3.
        # The output file path.
        command = [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-c:v', 'h264_cuvid',
            '-i', video_file,
            '-vn',
            '-acodec', 'libmp3lame',
            output_file
        ]

        # Use shlex.quote to handle file paths with spaces correctly.
        quoted_command = [shlex.quote(arg) for arg in command]

        print(f"Converting '{video_file}' to '{output_file}'...")
        try:
            # Execute the command. The 'check=True' will raise an exception
            # if the command returns a non-zero exit code.
            subprocess.run(
                quoted_command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            print(f"Successfully converted '{video_file}'.")
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion of '{video_file}':")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print("Error: 'ffmpeg' command not found. Please ensure ffmpeg is installed and in your system's PATH.")
            return

    print("All conversions complete!")

# This makes the script runnable directly.
if __name__ == "__main__":
    convert_videos_to_audio()
