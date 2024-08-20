import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import nibabel as nib
import albumentations as A

# Import classes from the EDA code
class ImageReader:
    def __init__(
        self, root:str, img_size:int=256,
        normalize:bool=False, single_class:bool=False
    ) -> None:
        pad_size = 256 if img_size > 256 else 224
        self.resize = A.Compose(
            [
                A.PadIfNeeded(min_height=pad_size, min_width=pad_size, value=0),
                A.Resize(img_size, img_size)
            ]
        )
        self.normalize = normalize
        self.single_class = single_class
        self.root = root

    def read_file(self, path:str) -> dict:
        scan_type = path.split('_')[-1]
        raw_image = nib.load(path).get_fdata()
        raw_mask = nib.load(path.replace(scan_type, 'seg.nii')).get_fdata()
        processed_frames, processed_masks = [], []
        for frame_idx in range(raw_image.shape[2]):
            frame = raw_image[:, :, frame_idx]
            mask = raw_mask[:, :, frame_idx]
            resized = self.resize(image=frame, mask=mask)
            processed_frames.append(resized['image'])
            processed_masks.append(
                1 * (resized['mask'] > 0) if self.single_class else resized['mask']
            )
        scan_data = np.stack(processed_frames, 0)
        if self.normalize:
            if scan_data.max() > 0:
                scan_data = scan_data / scan_data.max()
            scan_data = scan_data.astype(np.float32)
        return {
            'scan': scan_data,
            'segmentation': np.stack(processed_masks, 0),
            'orig_shape': raw_image.shape
        }

    def load_patient_scans(self, idx:int) -> dict:
        patient_id = str(idx).zfill(5)
        patient_dir = f'/content/MICCAI_BraTS_2018_Data_Training/HGG'
        scans = {}
        for folder_name in os.listdir(patient_dir):
            modality_dir = os.path.join(patient_dir, folder_name)
            if os.path.isdir(modality_dir):
                for file_name in os.listdir(modality_dir):
                    if file_name.endswith('.nii'):
                        scan_type = file_name.split('_')[-1].replace('.nii', '')
                        scan_filename = os.path.join(modality_dir, file_name)
                        scans[scan_type] = self.read_file(scan_filename)
        return scans

class ImageViewer3d:
    def __init__(
        self, reader:ImageReader,
        mri_downsample:int=10, mri_colorscale:str='Ice'
    ) -> None:
        self.reader = reader
        self.mri_downsample = mri_downsample
        self.mri_colorscale = mri_colorscale

    def load_clean_mri(self, image:np.array, orig_dim:int) -> dict:
        shape_offset = image.shape[1] / orig_dim
        z, x, y = (image > 0).nonzero()
        x, y, z = x[::self.mri_downsample], y[::self.mri_downsample], z[::self.mri_downsample]
        colors = image[z, x, y]
        return dict(x=x / shape_offset, y=y / shape_offset, z=z, colors=colors)

    def load_tumor_segmentation(self, image:np.array, orig_dim:int) -> dict:
        tumors = {}
        shape_offset = image.shape[1] / orig_dim
        sampling = {
            1: 1, 2: 3, 4: 5
        }
        for class_idx in sampling:
            z, x, y = (image == class_idx).nonzero()
            x, y, z = x[::sampling[class_idx]], y[::sampling[class_idx]], z[::sampling[class_idx]]
            tumors[class_idx] = dict(
                x=x / shape_offset, y=y / shape_offset, z=z,
                colors=class_idx / 4
            )
        return tumors

    def collect_patient_data(self, scan:dict) -> tuple:
        clean_mri = self.load_clean_mri(scan['scan'], scan['orig_shape'][0])
        tumors = self.load_tumor_segmentation(scan['segmentation'], scan['orig_shape'][0])
        markers_created = clean_mri['x'].shape[0] + sum(
            tumors[class_idx]['x'].shape[0] for class_idx in tumors
        )
        return [
            generate_3d_scatter(
                **clean_mri, scale=self.mri_colorscale, opacity=0.4,
                hover='skip', name='Brain MRI'
            ),
            generate_3d_scatter(
                **tumors[1], opacity=0.8,
                hover='all', name='Necrotic tumor core'
            ),
            generate_3d_scatter(
                **tumors[2], opacity=0.4,
                hover='all', name='Peritumoral invaded tissue'
            ),
            generate_3d_scatter(
                **tumors[4], opacity=0.4,
                hover='all', name='GD-enhancing tumor'
            ),
        ], markers_created

    def get_3d_scan(self, patient_idx:int, scan_type:str='flair') -> go.Figure:
        scan = self.reader.load_patient_scans(patient_idx)[scan_type]
        data, num_markers = self.collect_patient_data(scan)
        fig = go.Figure(data=data)
        fig.update_layout(
            title=f"[Patient id:{patient_idx}] brain MRI scan ({num_markers} points)",
            legend_title="Pixel class (click to enable/disable)",
            font=dict(
                family="Courier New, monospace",
                size=14,
            ),
            margin=dict(
                l=0, r=0, b=0, t=30
            ),
            legend=dict(itemsizing='constant')
        )
        return fig

# Helper function for generating a 3D scatter plot
def generate_3d_scatter(
    x: np.array, y: np.array, z: np.array, colors: np.array,
    size: int = 3, opacity: float = 0.2, scale: str = 'Teal',
    hover: str = 'skip', name: str = 'MRI'
) -> go.Scatter3d:
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers', hoverinfo=hover,
        marker=dict(
            size=size, opacity=opacity,
            color=colors, colorscale=scale
        ),
        name=name
    )

def main():
    st.title("Brain MRI 3D-EDA Visualization App")
    st.write("Input the patient ID to view their specific brain MRI scan.")
    
    # Get user input for patient_id and scan_type
    patient_id = st.number_input("Enter the patient ID:", min_value=0, step=1, value=0)
    scan_type = st.selectbox("Select the scan type:", options=["flair", "t1", "t2", "t1ce"])

    # Initialize ImageReader and ImageViewer3d
    root_dir = "/content/MICCAI_BraTS_2018_Data_Training"
    image_reader = ImageReader(root=root_dir, img_size=256, normalize=True, single_class=True)
    viewer = ImageViewer3d(image_reader, mri_downsample=10)

    # Get 3D scan based on user input
    if st.button("Display MRI Scan"):
        fig = viewer.get_3d_scan(patient_id, scan_type)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
