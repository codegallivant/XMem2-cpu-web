import os
import zipfile
import re
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file
from inference.run_on_video import run_on_video
import shutil
from flask import send_file


app = Flask(__name__, template_folder="flask-app-utils/templates")

# Configure upload folders
app.config["UPLOAD_FOLDER_DATA"] = "flask-app-utils/uploads/data"
app.config["UPLOAD_FOLDER_VIDEOS"] = "flask-app-utils/uploads/videos"
app.config["OUTPUT_FOLDER"] = "flask-app-utils/output"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB limit
app.secret_key = "supersecretkey"  # Needed for flash messages

# Ensure required directories exist
os.makedirs(app.config["UPLOAD_FOLDER_DATA"], exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER_VIDEOS"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)


def extract_zip(zip_path, extract_to):
    """Extracts a ZIP file to a given directory."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)

        file = request.files["file"]
        upload_type = request.form.get("upload_type")

        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)

        if upload_type == "data" and file.filename.endswith(".zip"):
            save_path = os.path.join(app.config["UPLOAD_FOLDER_DATA"], file.filename)
            file.save(save_path)
            extract_zip(save_path, app.config["UPLOAD_FOLDER_DATA"])
            os.remove(save_path)  # Delete the original ZIP after extraction
            flash("Data uploaded and extracted successfully!", "success")

        elif upload_type == "video":
            save_path = os.path.join(app.config["UPLOAD_FOLDER_VIDEOS"], "video.mp4")
            file.save(save_path)
            flash("Video uploaded successfully!", "success")

        else:
            flash("Invalid file type", "error")

    return render_template("index.html")


@app.route("/inference", methods=["GET"])
def infer():
    """Runs inference, zips the output folder, and returns the zip file."""
    masks_path = os.path.join(app.config["UPLOAD_FOLDER_DATA"], "Annotations")
    output_path = app.config["OUTPUT_FOLDER"]
    video_path = os.path.join(app.config["UPLOAD_FOLDER_VIDEOS"], "video.mp4")
    zip_path = os.path.join(app.config["UPLOAD_FOLDER_DATA"], "output.zip")

    if not os.path.exists(masks_path) or not os.path.exists(video_path):
        return jsonify({"error": "Missing required files"}), 400

    frames_with_masks = []
    for file_path in (p for p in Path(masks_path).iterdir() if p.is_file()):
        frame_number_match = re.search(r"\d+", file_path.stem)
        if frame_number_match is None:
            return jsonify({"error": f"Invalid mask file: {file_path}"}), 400
        frames_with_masks.append(int(frame_number_match.group()))

    print("Using masks for frames:", frames_with_masks)

    os.makedirs(output_path, exist_ok=True)
    run_on_video(video_path, masks_path, output_path, frames_with_masks)

    if not os.path.exists(output_path) or not os.listdir(output_path):
        return jsonify({"error": "Inference failed or no output generated"}), 500

    # Create a zip archive of the output folder
    shutil.make_archive(zip_path[:-4], 'zip', output_path)

    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
