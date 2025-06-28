import os
import zipfile
import re
import json
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file, session
from inference.run_on_video import run_on_video
from maskandpoint import points_to_mask, mask_to_points
import shutil
import cv2
import numpy as np


app = Flask(__name__, template_folder="flask-app-utils/templates")

# Configure upload folders
app.config["UPLOAD_FOLDER_DATA"] = "flask-app-utils/uploads/data"
app.config["UPLOAD_FOLDER_VIDEOS"] = "flask-app-utils/uploads/videos"
app.config["OUTPUT_FOLDER"] = "flask-app-utils/output"
app.config["TEMP_FOLDER"] = "flask-app-utils/temp"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB limit
app.secret_key = "supersecretkey"  # Needed for flash messages

# Ensure required directories exist
os.makedirs(app.config["UPLOAD_FOLDER_DATA"], exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER_VIDEOS"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
os.makedirs(app.config["TEMP_FOLDER"], exist_ok=True)


def extract_zip(zip_path, extract_to):
    """Extracts a ZIP file to a given directory without keeping the top folder."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        
        # Find the common top-level folder, if any
        top_folder = os.path.commonprefix(members).rstrip('/')
        
        for member in members:
            # Strip the top-level folder path
            target_path = os.path.join(extract_to, os.path.relpath(member, top_folder)) if top_folder else os.path.join(extract_to, member)
            
            if member.endswith('/'):
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())


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


def get_frames_with_masks(masks_path):
    frames_with_masks = []
    for file_path in (p for p in Path(masks_path).iterdir() if p.is_file() and os.path.splitext(p)[1] == '.png'):
        frame_number_match = re.search(r"\d+", file_path.stem)
        if frame_number_match is None:
            # Fix: Skip invalid files instead of returning a response object
            continue
        frames_with_masks.append(int(frame_number_match.group()))
    return frames_with_masks

@app.route("/inference", methods=["GET"])
def infer():
    """Runs inference, zips the output folder, and returns the zip file."""
    masks_path = app.config["UPLOAD_FOLDER_DATA"]
    output_path = app.config["OUTPUT_FOLDER"]
    video_path = os.path.join(app.config["UPLOAD_FOLDER_VIDEOS"], "video.mp4")
    
    # Fix: Create the zip in the output folder instead of uploads/data
    zip_base = os.path.join(app.config["OUTPUT_FOLDER"], "output")
    zip_path = f"{zip_base}.zip"

    if not os.path.exists(masks_path) or not os.path.exists(video_path):
        return jsonify({"error": "Missing required files"}), 400

    try:
        frames_with_masks = get_frames_with_masks(masks_path)
        if not frames_with_masks:
            return jsonify({"error": "No valid mask files found"}), 400
            
        print("Using masks for frames:", frames_with_masks)

        os.makedirs(output_path, exist_ok=True)
        run_on_video(video_path, masks_path, output_path, frames_with_masks)

        if not os.path.exists(output_path) or not os.listdir(output_path):
            return jsonify({"error": "Inference failed or no output generated"}), 500

        # Fix: Use zip_base for the archive name
        shutil.make_archive(zip_base, 'zip', output_path)

        # Check if the zip was created
        if not os.path.exists(zip_path):
            return jsonify({"error": "Failed to create output zip file"}), 500

        return send_file(zip_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Error during inference: {str(e)}"}), 500


@app.route("/examine", methods=["GET"])
def examine():
    """Renders the examine page for frame-by-frame video inspection"""
    video_path = os.path.join(app.config["UPLOAD_FOLDER_VIDEOS"], "video.mp4")
    
    if not os.path.exists(video_path):
        flash("No video has been uploaded", "error")
        return redirect(url_for("index"))
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        flash("Could not open video file", "error")
        return redirect(url_for("index"))
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return render_template("examine.html", total_frames=total_frames, fps=fps, width=width, height=height)


@app.route("/get_frame/<int:frame_num>", methods=["GET"])
def get_frame(frame_num):
    """Extracts and returns a specific frame from the video"""
    video_path = os.path.join(app.config["UPLOAD_FOLDER_VIDEOS"], "video.mp4")
    frame_path = os.path.join(app.config["TEMP_FOLDER"], f"frame_{frame_num}.jpg")
    
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video file"}), 500
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": f"Could not read frame {frame_num}"}), 404
    
    cv2.imwrite(frame_path, frame)
    
    return send_file(frame_path, mimetype='image/jpeg')


@app.route("/infer_frame/<int:frame_num>", methods=["GET"])
def infer_frame(frame_num):
    """Infers a mask for a single frame"""
    video_path = os.path.join(app.config["UPLOAD_FOLDER_VIDEOS"], "video.mp4")
    masks_path = app.config["UPLOAD_FOLDER_DATA"]
    output_path = os.path.join(app.config["TEMP_FOLDER"], f"inference_{frame_num}")
    frame_video_path = os.path.join(app.config["TEMP_FOLDER"], f"frame_{frame_num}.mp4")
    mask_output_path = os.path.join(output_path, "masks", f"frame_{frame_num:06d}.png")
    
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video file"}), 500
    
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    try:
        os.makedirs(output_path, exist_ok=True)
        out = cv2.VideoWriter(frame_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
        
        # Write all frames up to the current frame
        for i in range(frame_num + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                out.release()
                return jsonify({"error": f"Could not read frame {i}"}), 404
            out.write(frame)
        
        cap.release()
        out.release()

        if not os.path.exists(masks_path):
            return jsonify({"error": "Missing mask data directory"}), 400

        frames_with_masks = get_frames_with_masks(masks_path)

        if frame_num == 0:
            frames_with_masks = [0]
        else:
            frames_with_masks = [frame_num-1]
        print("Using masks for frames:", frames_with_masks)
        
        # Run inference on the video up to the current frame
        run_on_video(frame_video_path, masks_path, output_path, frames_with_masks)
        
        # Check if mask was generated
        if not os.path.exists(mask_output_path):
            # If no specific mask was generated, create a blank one
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return jsonify({"error": f"Could not read frame {frame_num}"}), 404
                
            blank_mask = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            cv2.imwrite(mask_output_path, blank_mask)
        
        mask_path = os.path.join(app.config["UPLOAD_FOLDER_DATA"], f"frame_{frame_num:06d}.png")
        cv2.imwrite(mask_path, cv2.imread(mask_output_path))

        return send_file(mask_output_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Error during frame inference: {str(e)}"}), 500
        

# Fix for app.py - Update the get_mask_points route

@app.route("/get_mask_points/<int:frame_num>", methods=["GET"])
def get_mask_points(frame_num):
    """Converts a mask to point data for editing"""
    # Look first for a generated mask from inference
    mask_path = os.path.join(app.config["TEMP_FOLDER"], f"inference_{frame_num}", "masks", f"frame_{frame_num:06d}.png")
    
    # If that doesn't exist, check for a user-created mask
    if not os.path.exists(mask_path):
        mask_path = os.path.join(app.config["UPLOAD_FOLDER_DATA"], f"frame_{frame_num:06d}.png")
    
    # If still no mask, check for older naming convention
    if not os.path.exists(mask_path):
        mask_path = os.path.join(app.config["UPLOAD_FOLDER_DATA"], f"mask_{frame_num}.png")
    
    if not os.path.exists(mask_path):
        # If no mask exists, create a blank one to start editing
        video_path = os.path.join(app.config["UPLOAD_FOLDER_VIDEOS"], "video.mp4")
        if not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 404
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video file"}), 500
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"error": f"Could not read frame {frame_num}"}), 404
            
        temp_mask_path = os.path.join(app.config["TEMP_FOLDER"], f"temp_mask_{frame_num}.png")
        blank_mask = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        cv2.imwrite(temp_mask_path, blank_mask)
        mask_path = temp_mask_path
    
    points_path = os.path.join(app.config["TEMP_FOLDER"], f"points_{frame_num}.json")
    
    try:
        result = mask_to_points(mask_path, points_path)
        # Add debug info to response
        if not result.get("shapes"):
            result["shapes"] = []
            print("Warning: No shapes extracted from mask")
        
        print(f"Extracted {len(result.get('shapes', []))} shapes from mask")
        return jsonify(result)
    except Exception as e:
        print(f"Error converting mask to points: {str(e)}")
        # Return empty shape data instead of error
        return jsonify({"shapes": []})


# Modification for app.py
# Update the save_edited_mask route to ensure masks are properly saved

@app.route("/save_edited_mask", methods=["POST"])
def save_edited_mask():
    """Saves edited point data as a mask and adds to training data"""
    try:
        # Get the data from the request
        data = request.json
        if not data:
            print("Error: No JSON data received")
            return jsonify({"error": "No data received"}), 400
            
        frame_num = data.get("frame_num")
        points_data = data.get("points_data")
        
        # Debug logging
        print(f"Received data for frame {frame_num}")
        print(f"Points data: {points_data}")
        
        if frame_num is None or points_data is None:
            print(f"Missing data: frame_num={frame_num}, points_data_type={type(points_data)}")
            return jsonify({"error": "Missing required data"}), 400
        
        # Get video dimensions for mask generation
        video_path = os.path.join(app.config["UPLOAD_FOLDER_VIDEOS"], "video.mp4")
        
        if not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 404
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video file"}), 500
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        cap.release()
        
        # Generate standard filename format for mask
        mask_path = os.path.join(app.config["UPLOAD_FOLDER_DATA"], f"frame_{frame_num:06d}.png")
        
        # Ensure data directory exists
        os.makedirs(app.config["UPLOAD_FOLDER_DATA"], exist_ok=True)
        
        # Generate mask from points
        points_to_mask(points_data, mask_path, frame_size)
        
        # Save frame for reference (optional)
        frame_path = os.path.join(app.config["TEMP_FOLDER"], f"frame_{frame_num}.jpg")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            cv2.imwrite(frame_path, frame)
        
        return jsonify({"success": True, "message": f"Mask saved for frame {frame_num}"})
    except Exception as e:
        import traceback
        print(f"Error in save_edited_mask: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)