import React from "react";
import {Card, Container} from "react-bootstrap";

export const Training = () => {
    return (
        <div className={'d-flex justify-content-center flex-column'}>
            <Container className={'mt-3 ms-1 row d-flex align-items-center justify-content-center'} fluid>
                <Container className={'col-lg-8 col-sm-12 rounded'}>
                    <Card className={'border'} style={{height: "60px"}}>
                        <h1 className={'text-center'}>Training YOLO Models for detection</h1>
                    </Card>
                </Container>
            </Container>
        </div>
    );
}