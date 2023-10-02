import { Route, Routes, BrowserRouter, Link } from 'react-router-dom'
import './App.css';
import { Login, Dashboard, Case } from './components'

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route
                    path="/"
                    element={<Login />}
                />
                <Route
                    path="/main/dashboard"
                    element={<Dashboard />}
                />
                <Route
                    path="/auth/login"
                    element={<Login />}
                />
                <Route
                    path="/main/case"
                    element={<Case />}
                />
            </Routes>
        </BrowserRouter>
    );
}

export default App;
