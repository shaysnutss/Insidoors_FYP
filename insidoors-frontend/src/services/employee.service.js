import axios from "axios";
import authHeader from "./auth-header";

const API_URL = "http://localhost:30000/api/v1/accountByBA/";

class employeeService {
  viewAllEmployees = () => {
    return axios.get(API_URL + "viewAllEmployees", { headers: authHeader()});
  };

  viewAllTasksByEmployee = (id) => {
    return axios.get(API_URL + "viewIncidentsByEmployeeId/" + id, {headers: authHeader()} );
  }

  viewEmployeeByName = (name) => {
    return axios.get(API_URL + "viewEmployeeByName/" + name, {headers: authHeader()} );
  }

  viewEmployeeByName = (name) => {
    return axios.get(API_URL + "viewEmployeeByName/" + name, {headers: authHeader()} );
  }

}

export default new employeeService();