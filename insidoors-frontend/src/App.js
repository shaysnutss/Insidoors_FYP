import { Route, Routes, BrowserRouter } from 'react-router-dom'
import './App.css';
import { Login, Dashboard, Case, Employees } from './components'

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
                <Route
                    path="/main/employees"
                    element={<Employees />}
                />
                
            </Routes>
        </BrowserRouter>
    );
}

export default App;
