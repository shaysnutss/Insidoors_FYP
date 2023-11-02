import axios from "axios";
import authHeader from "./auth-header";

const API_URL = "http://localhost:30000/api/v1/";

class userService {
  getAccountById = () => {
    
    return axios.get(API_URL + "demo-controller", { headers: authHeader() });
  };

  getAllCases = () => {
    //return axios.get("https://jsonplaceholder.typicode.com/todos/");
    return axios.get("http://localhost:30010/api/v1/viewAllTasks")
  }

  getSingleCase = (id) => {
    return axios.get("http://localhost:30010/api/v1/viewTaskById/" + id)
  }


}

export default new userService();