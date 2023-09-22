import axios from "axios";
import authHeader from "./auth-header";

const API_URL = "http://localhost:8080/api/v1/";

class userService{
  getAccountById = () => {
    return axios.get(API_URL + "demo-controller", { headers: authHeader() });
  };
}

export default new userService();