import axios from "axios";
import authHeader from "./auth-header";

const API_URL = "http://localhost:8080/api/v1/";

class userService{
  getAccountById = () => {
    return axios.get(API_URL + "demo-controller", { headers: authHeader() });
  };

  getAllCases = () => {
    return axios.get("https://jsonplaceholder.typicode.com/todos/")
  // return axios.get("http://localhost:30000/api/v1/accountByTasks/viewAllTasks", { headers: authHeader() })

  }

  
}

export default new userService();