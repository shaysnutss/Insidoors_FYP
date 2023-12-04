package com.service.pcaccess;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

@Repository
public interface PcAccessRepository extends JpaRepository<PcAccess, Long> {
    
    @Query(value = "SELECT * FROM pc_access WHERE ID > ?1", nativeQuery = true)
    List<PcAccess> findAllAfterId(Long desiredId);
}
