import { Route, Routes, BrowserRouter } from 'react-router-dom'
import './App.css';
import { Login, Register } from './components'

function App() {
  return (
      <BrowserRouter>
          <Routes>
              <Route
                  path="/"
                  element={ <Login /> }
              />
              <Route
                  path="/register"
                  element={ <Register /> }
              />
              <Route
                  path="/login"
                  element={ <Login /> }
              />
          </Routes>
      </BrowserRouter>
  );
}

export default App;
