import os
import subprocess

def convert_mp4_to_mp3(input_directory, output_directory):
    """
    Converts all MP4 files in the input_directory to MP3 files
    and saves them in the output_directory.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    mp4_files = [f for f in os.listdir(input_directory) if f.endswith('.mp4')]
    
    if not mp4_files:
        print(f"No MP4 files found in {input_directory}")
        return

    print(f"Found {len(mp4_files)} MP4 files. Starting conversion...")

    for mp4_file in mp4_files:
        input_filepath = os.path.join(input_directory, mp4_file)
        # Change file extension from .mp4 to .mp3
        output_filename = os.path.splitext(mp4_file)[0] + '.mp3'
        output_filepath = os.path.join(output_directory, output_filename)

        print(f"Converting '{mp4_file}' to '{output_filename}'...")
        
        try:
            # ffmpeg command to convert video to audio
            # -i: input file
            # -vn: disable video recording (only extract audio)
            # -ab 192k: set audio bitrate to 192kbps (common quality)
            # -map_metadata 0: copy metadata from input to output
            # -hide_banner: hide ffmpeg startup banner
            # -loglevel warning: suppress verbose output, only show warnings/errors
            command = [
                'ffmpeg',
                '-i', input_filepath,
                '-vn',
                '-ab', '192k',
                '-map_metadata', '0',
                '-y',  # Overwrite output files without asking
                output_filepath
            ]
            
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully converted '{mp4_file}'")
        except subprocess.CalledProcessError as e:
            print(f"Error converting '{mp4_file}':")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
        except Exception as e:
            print(f"An unexpected error occurred during conversion of '{mp4_file}': {e}")

    print("Conversion process complete! 🎉")

if __name__ == "__main__":
    # Define your input and output directories
    # Assuming video files are in the same directory as the script,
    # or specify the full path.
    input_video_directory = '.'  # Current directory, or change to '/path/to/your/videos'
    output_audio_directory = 'audio'

    convert_mp4_to_mp3(input_video_directory, output_audio_directory)
