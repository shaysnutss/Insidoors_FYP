package com.service.behavioralanalysisservice;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface BehavioralAnalysisServiceRepository extends JpaRepository<BehavioralAnalysisService, Long> {
    
    BehavioralAnalysisService findByEmployeeId(int employeeId);
    boolean existsByEmployeeId(Long id);

}
