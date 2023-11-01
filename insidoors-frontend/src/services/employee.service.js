import axios from "axios";
import authHeader from "./auth-header";

const API_URL = "http://localhost:30011/api/v1/BAComposite/";

class employeeService {
  viewAllEmployees = () => {
    return axios.get(API_URL + "viewAllEmployees", { });
  };

  viewAllTasksByEmployee = (id) => {
    return axios.get(API_URL + "viewIncidentsByEmployeeId/" + id );

  }




}

export default new employeeService();