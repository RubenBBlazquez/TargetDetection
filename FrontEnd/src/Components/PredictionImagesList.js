import React from "react";
import {Card, Container} from "react-bootstrap";
import {GET_PREDICTION_IMAGES_MODES} from "../Utils/UtilsPredictions";
import CustomModal from "./modal";

export default class PredictionImagesList extends React.PureComponent {
    constructor(props) {
        super(props);

        this.state = {
            show: false,
            onHide: () => this.setState({show: false}),
            selectedImage: 0,
            images: this.get_images()
        }
    }

    get_images = () => {
        const {mode} = this.props;

        switch (mode) {
            case GET_PREDICTION_IMAGES_MODES.ALL:
                return;
            case GET_PREDICTION_IMAGES_MODES.ONLY_PREDICTION:
                return;
            case GET_PREDICTION_IMAGES_MODES.ONLY_NOT_PREDICTION:
                return;
            case GET_PREDICTION_IMAGES_MODES.LASTS:
                return;
            default:
                return [
                    {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    }, {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    }, {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    }, {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    }, {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    }, {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    }, {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    }, {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    }, {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    },
                    {
                        'original_image': 'non_target_image.jpg',
                        'prediction_image': 'target_image.jpg',
                        'prediction_image_shots': 'target_image_2.jpg',
                    }
                ]
        }

    }

    getSelectedImageModalInfo = () => {
        const {selectedImage, images, show, onHide} = this.state;

        if (!show) {
            return {}
        }

        const imageInformation = images[selectedImage]

        return {
            title: 'Prediction Information',
            body:
                <Container>
                    <Container className={'row'}>
                        <Container className={'d-flex flex-column align-items-center col-6'}>
                            <h3>Original Image</h3>
                            <img className={'w-100 h-100 img-fluid img-responsive rounded border'}
                                 alt={imageInformation['original_image']}
                                 src={imageInformation['original_image']}/>
                        </Container>
                        <Container className={'d-flex flex-column align-items-center col-6'}>
                            <h3>Predicted Target</h3>
                            <img className={'w-100 h-100 img-fluid img-responsive rounded border'}
                                 alt={imageInformation['prediction_image']}
                                 src={imageInformation['prediction_image']}/>
                        </Container>
                    </Container>
                    <Container className={'row mt-4'}>
                        <Container className={'d-flex flex-column align-items-center col-8'}>
                            <h3>Predicted Shots</h3>
                            <img className={'w-100 h-100 img-fluid img-responsive rounded border'}
                                 alt={imageInformation['prediction_image_shots']}
                                 src={imageInformation['prediction_image_shots']}/>
                        </Container>
                    </Container>
                </Container>,
            footer: null,
            show,
            onHide
        }
    }

    render() {
        const {images} = this.state;


        return (
            <>
                <CustomModal {...this.getSelectedImageModalInfo()}/>
                <Container className={'border rounded mt-3 w-100 p-2 d-flex justify-content-center position-relative'}
                           fluid>
                    <Container className={'overflow-x-auto d-flex w-100'} fluid>
                        {
                            images.map((image, index) => {
                                image = image['original_image'];

                                return (
                                    <Card key={`imagePrediction-${index}`} className={'col-1 m-1'}
                                          onClick={() => this.setState({selectedImage: index, show: true})}>
                                        <img className={'w-100 h-100 img-fluid img-responsive rounded border'}
                                             alt={image} src={image}/>
                                    </Card>
                                )
                            })
                        }
                    </Container>
                </Container>
            </>

        );
    }
}