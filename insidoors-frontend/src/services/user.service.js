import axios from "axios";
import authHeader from "./auth-header";

const API_URL = "http://localhost:30000/api/v1/account/";

class userService{
  getAccountById = () => {
    return axios.get(API_URL + "demo-controller", { headers: authHeader() });
  };

  getAllCases = () => {
    return axios.get("http://localhost:30000/api/v1/accountByTasks/viewAllTasks", { headers: authHeader() })
  }

  
}

export default new userService();