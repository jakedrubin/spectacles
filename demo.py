"""demo.py
Spectacles Recommendation Web Demo

A Gradio-based web interface for the face-to-frame recommendation system.
Users can upload face images and receive personalized glasses recommendations.
"""
import gradio as gr
from pathlib import Path
import pandas as pd
from src.face_analysis import extract_face_features
from src.recommend import recommend


def recommendation_pipeline(image):
    """
    End-to-end pipeline: Face Analysis → Glasses Recommendation
    
    Args:
        image: Uploaded image (numpy array from Gradio)
        
    Returns:
        Tuple of (analysis_text, recommendation_html)
    """
    if image is None:
        return "Please upload an image.", ""
    
    # Save temporary image
    temp_path = Path("temp_upload.jpg")
    import cv2
    cv2.imwrite(str(temp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    try:
        # Step A: Face Analysis
        face_features = extract_face_features(str(temp_path))
        
        if face_features is None:
            return "❌ **Error: No face detected in the image.**\n\nPlease upload a clear photo with a visible face.", ""
        
        # Format facial features for display
        analysis_text = "### ✅ Face Detected Successfully!\n\n"
        analysis_text += "**Extracted Facial Features:**\n\n"
        
        # Debug: print features to console
        print(f"\n[DEBUG] Extracted face features: {face_features}")
        
        for key, value in face_features.items():
            # Make it more readable
            readable_key = key.replace('Ratio', ' Ratio').replace('Distance', ' Distance').replace('Deviation', ' Deviation')
            analysis_text += f"- **{readable_key}**: {value:.4f}\n"
        
        # Step B: Glasses Recommendation
        model_path = Path("models/regressor.joblib")
        if not model_path.exists():
            return analysis_text, "\n\n❌ **Error: Model not found.** Please ensure `models/regressor.joblib` exists."
        
        try:
            recommendations = recommend(face_features, model_path, top_k=5)
            
            # Format recommendations for display
            rec_html = "\n\n### 🕶️ **Top 5 Recommended Frames:**\n\n"
            
            for idx, row in recommendations.iterrows():
                rec_html += f"**{idx + 1}.** "
                
                # Add available fields
                if 'FrameID' in row:
                    rec_html += f"Frame ID: **{row['FrameID']}**"
                
                details = []
                if 'Brand' in row and pd.notna(row['Brand']):
                    details.append(f"Brand: {row['Brand']}")
                if 'Shape' in row and pd.notna(row['Shape']):
                    details.append(f"Shape: {row['Shape']}")
                if 'Color' in row and pd.notna(row['Color']):
                    details.append(f"Color: {row['Color']}")
                if 'PredictedBeautyScore' in row:
                    details.append(f"Beauty Score: {row['PredictedBeautyScore']:.3f}")
                
                if details:
                    rec_html += " | " + " | ".join(details)
                
                rec_html += "\n\n"
            
            result_text = analysis_text + rec_html
            return result_text, ""
            
        except Exception as e:
            return analysis_text, f"\n\n❌ **Recommendation Error:** {str(e)}"
    
    except Exception as e:
        return f"❌ **Error during processing:** {str(e)}", ""
    
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


def create_demo():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🕶️ Spectacles: Face-to-Frame Recommendation Demo
            
            Upload a photo of your face to get personalized glasses recommendations based on your unique facial features.
            
            ### How it works:
            1. **Upload** a clear photo of your face
            2. **Analyze** - Our AI extracts your facial features
            3. **Recommend** - Get the top 5 frames that complement your face shape
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Your Photo",
                    type="numpy",
                    sources=["upload", "webcam"],
                    height=400
                )
                
                submit_btn = gr.Button(
                    "🔍 Get Recommendation",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    **Tips for best results:**
                    - Use a well-lit, front-facing photo
                    - Ensure your entire face is visible
                    - Remove existing glasses if possible
                    - Avoid heavy filters or makeup
                    """
                )
            
            with gr.Column(scale=1):
                output_text = gr.Markdown(
                    label="Results",
                    value="Upload an image and click 'Get Recommendation' to see results."
                )
                
                error_text = gr.Markdown(visible=False)
        
        # Connect the button
        submit_btn.click(
            fn=recommendation_pipeline,
            inputs=[image_input],
            outputs=[output_text, error_text]
        )
        
        gr.Markdown(
            """
            ---
            
            ### About
            This demo uses MediaPipe for facial landmark detection and a trained Random Forest model 
            to predict which frames will best complement your facial features.
            
            **Facial metrics analyzed:** Symmetry, Golden Ratio Deviation, Eye Spacing, Jawline Width, 
            Brow-to-Eye Distance, and Lip-to-Nose Distance.
            """
        )
    
    return demo


def main():
    """Launch the Gradio demo."""
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == '__main__':
    main()
