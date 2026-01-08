# SAM UI

A webui that uses SAM3 to segment images and output segmentation masks.

## Project structure

* Starting from totally greenfield
* monorepo
* uv as python package manager
  * src layout for python packages
* ruff as linter/formatter, both style and complexity flags active.

## Components

* webui
* sam3 segmentation
* blob storage
* database for processing history


## Technologies

* python
* Streamlit as webui framework
  * streamlit component for bbox drawing: https://github.com/blackary/streamlit-image-coordinates. rectangle_select example: https://image-coordinates.streamlit.app/rectangle_select
* FastAPI for as REST API framework
* pydantic v2
* SAM3: https://github.com/facebookresearch/sam3
  * examples
    * Image prediction: https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb
    * batch inference: https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_batched_inference.ipynb


## Webui

### Upload page

- Drag and drop images or select folder to upload.
- Tiled image view to see uploaded images

### Annotation page

- Large single image view for annotation purposes
- Tiled image view below larger image 
- Right panel with annotation information and possibility to delete
- each rectangle should be shown in different colors
- Navigation
  - clicking tiled image view
  - arrow keys to navigate to next image

### Processing page

- Large single image view over processed image
- Tiled image view of processed images
  - bbox drawn and segmentation overlay in the same image.
- Download button to download coco annotations in json file
- Navigation
  - clicking tiled image view
  - arrow keys to navigate to next image


## Deployment

- Deployment through docker compose
- postgres database for storing image- and processing information
- Azurite for image blob storage
- hard coded env variables establishing connection