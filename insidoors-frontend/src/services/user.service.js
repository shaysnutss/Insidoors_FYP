import axios from "axios";
import authHeader from "./auth-header";

const API_URL = "http://localhost:30000/api/v1/";

class userService {
  getAccountById = () => {
    return axios.get(API_URL + "demo-controller", { headers: authHeader() });
  };

  getName = () => {
    return axios.get(API_URL + "accountByTasks/viewName", { headers: authHeader() });
  }

  getUser = () => {
    return axios.get(API_URL + "accountByTasks/getAccount", { headers: authHeader() });
  }

  getAllCases = () => {
    //return axios.get("https://jsonplaceholder.typicode.com/todos/");
    return axios.get("http://localhost:30010/api/v1/viewAllTasks");
  }

  getCaseById = (id) => {
    //return axios.get("https://jsonplaceholder.typicode.com/todos/" + id );
    return axios.get("http://localhost:30010/api/v1/viewTaskById/" + id);
  }

  getAllCommentsById = (id) => {
    //return axios.get("https://jsonplaceholder.typicode.com/todos/" + id);
    return axios.get("http://localhost:30010/api/v1/viewAllCommentsByTaskId/" + id);
  }

  getAllSoc = () => {
    return axios.get("http://localhost:30010/api/v1/listSOCs");
  }

  putSoc = (id, accountId, status) => {
    return axios({
      method: 'put',
      url: 'http://localhost:30010/api/v1/assignSOC/' + id,
      data: {
        "accountId": accountId,
        "status": status
      }
    });
  }

  closeCase = (id, riskRating) => {
    return axios({
      method: 'put',
      url: 'http://localhost:30010/api/v1/closeCase/' + id,
      data: {
        "riskRating": riskRating,
        "truePositive": true
      }
    });
  }

  addComment = (id, socId, desc) => {
    return axios({
      method: 'post',
      url: 'http://localhost:30010/api/v1/addComment/' + id,
      data: {
        "accountId": socId,
        "commentDescription": desc,
        "taskManagementId": id
      }
    });
  }
}

export default new userService;