import React from "react";
import {Container, Form, InputGroup} from "react-bootstrap";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import {icon} from "@fortawesome/fontawesome-svg-core/import.macro";

export default class Filters extends React.PureComponent {
    render() {
        return (
            <div className={'rounded row p-2 d-flex justify-content-center'}>
                <Container className={'col-4'}>
                    <InputGroup>
                        <InputGroup.Text id="basic-addon1">
                            <FontAwesomeIcon className='p-1'
                                             icon={icon({name: 'calendar'})}/> </InputGroup.Text>
                        <Form.Control type="date"/>
                    </InputGroup>
                </Container>
                <Container className={'col-4'}>
                    <Form.Control type="date"/>
                </Container>
                <Container className={'col-4'}>
                    <Form.Control type="date"/>
                </Container>
            </div>
        );
    }
}