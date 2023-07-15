import React from "react";
import Container from 'react-bootstrap/Container';
import Navbar from 'react-bootstrap/Navbar';

function Home() {
    return (
        <div style={{height: "100vh"}} className='container-fluid position-relative w-100 border border-dark d-flex justify-content-center align-items-center'>
            <div className='w-50 mb-5'>
                <Navbar className="bg-body-tertiary">
                    <Container>
                        <Navbar.Brand href="#home">Brand link</Navbar.Brand>
                    </Container>
                </Navbar>
                <br/>
                <Navbar className="bg-body-tertiary">
                    <Container>
                        <Navbar.Brand>Brand text</Navbar.Brand>
                    </Container>
                </Navbar>
                <br/>
                <Navbar className="bg-body-tertiary">
                    <Container>
                        <Navbar.Brand href="#home">
                            <img
                                src="/img/logo.svg"
                                width="30"
                                height="30"
                                className="d-inline-block align-top"
                                alt="React Bootstrap logo"
                            />
                        </Navbar.Brand>
                    </Container>
                </Navbar>
                <br/>
                <Navbar className="bg-body-tertiary">
                    <Container>
                        <Navbar.Brand href="#home">
                            <img
                                alt=""
                                src="/img/logo.svg"
                                width="30"
                                height="30"
                                className="d-inline-block align-top"
                            />{' '}
                            React Bootstrap
                        </Navbar.Brand>
                    </Container>
                </Navbar>
            </div>
        </div>
    );
}

export default Home;