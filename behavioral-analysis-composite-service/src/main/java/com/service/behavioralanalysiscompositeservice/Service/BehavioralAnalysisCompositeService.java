package com.service.behavioralanalysiscompositeservice.Service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
/* The single purpose of this class is to contain helper methods for this service's APIs. */

public class BehavioralAnalysisCompositeService {


    // Sending GET request to employee service to get all employees
    public List<Map<String, Object>> getMethod(String baseURL){

        HttpClient client = HttpClients.createDefault();
        HttpGet request = new HttpGet(baseURL);
        List<Map<String, Object>> allResults = null;
        try{
            HttpResponse response = client.execute(request);
            if(response.getStatusLine().getStatusCode() == 404){
                return null;

            }else if(response.getStatusLine().getStatusCode() != 200){
                throw new RuntimeException("Failed : HTTP error code : "
                        + response.getStatusLine().getStatusCode());
            }
            HttpEntity entity = response.getEntity();
            String responseString = EntityUtils.toString(entity, "UTF-8");
            ObjectMapper mapper = new ObjectMapper();
            allResults = mapper.readValue(responseString, new TypeReference<List<Map<String, Object>>>(){});

        }catch(Exception e){
            e.printStackTrace();
            return null;
        }

        return allResults;

    }

    // Sending GET request to employee service to get all employees
    public Map<String, Object> getEmployeeMethod(String baseURL){

        HttpClient client = HttpClients.createDefault();
        HttpGet request = new HttpGet(baseURL);
        Map<String, Object> allResults = null;
        try{
            HttpResponse response = client.execute(request);
            if(response.getStatusLine().getStatusCode() == 404){
                return null;

            }else if(response.getStatusLine().getStatusCode() != 200){
                throw new RuntimeException("Failed : HTTP error code : "
                        + response.getStatusLine().getStatusCode());
            }
            HttpEntity entity = response.getEntity();
            String responseString = EntityUtils.toString(entity, "UTF-8");
            ObjectMapper mapper = new ObjectMapper();
            allResults = mapper.readValue(responseString, new TypeReference<Map<String, Object>>(){});

        }catch(Exception e){
            e.printStackTrace();
            return null;
        }

        return allResults;

    }

    /*
    This method is to help find the behaviour analysis of a particular employee.
    If found, a map of that behaviour analysis would be returned.
    If that employee has no behaviour analysis, meaning no incidents committed, null will be returned.
     */
    public Map<String,Object> employeeBH(int employeeId, List<Map<String, Object>> bhList){

        // We iterate through the list of all behaviour analysis
        for (Map<String, Object> bh : bhList){
            int bhEmployeeId = (int) bh.get("employeeId");
            // We are trying to ascertain if this bh belongs to the relevant employee
            if(bhEmployeeId == employeeId){
                // if so return that bh
                return bh;
            }
        }
        // if no bh found for the employee, we return null
        return null;
        /*
        Here, we choose to iterate through a list returned from one api call instead of sending
        an API call for each employee. This is because we want to minimise api calls and consequently
        DB calls as they increase lag.
         */
    }


}
