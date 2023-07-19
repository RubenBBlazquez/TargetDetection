import {Button} from "react-bootstrap";
import Modal from 'react-bootstrap/Modal';

export default function CustomModal(props) {
    const {title, body, footer, show, onHide} = props;

    return (
        <Modal
            show={show}
            size="lg"
            aria-labelledby="contained-modal-title-vcenter"
            centered
        >
            <Modal.Header className={'d-flex justify-content-center'}>
                <Modal.Title id="contained-modal-title-vcenter">
                    {title}
                </Modal.Title>
            </Modal.Header>
            <Modal.Body>
                {
                    body
                }
            </Modal.Body>
            {
                footer &&
                <Modal.Footer>
                    {
                        footer
                    }
                </Modal.Footer>
            }
            <Modal.Footer>
                <Button onClick={onHide}>Close</Button>
            </Modal.Footer>
        </Modal>
    );
}