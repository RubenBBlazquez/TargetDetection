import {BrowserRouter, Routes, Route, Link} from 'react-router-dom';
import Home from "./Home/Home";
import Predictions from "./Predictions/Predictions";
import {Nav} from 'react-bootstrap';

function App() {
        return (
            <div>
                <BrowserRouter>
                    <nav>
                        <Link to="/">Home</Link>
                        <Link to="/predictions">Predictions</Link>
                    </nav>
                    <Routes>
                        <Route path="/" element={<Home/>}/>
                        <Route path="/predictions" element={<Predictions/>}/>
                    </Routes>
                </BrowserRouter>
            </div>

        );
}

export default App;