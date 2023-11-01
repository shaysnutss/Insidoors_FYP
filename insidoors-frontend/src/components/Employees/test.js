import React, { useState, useEffect } from "react";
import "./Employees.css"
import { useNavigate } from "react-router-dom";
import Navigation from '../Navigation/Navigation';
import employeeService from "../../services/employee.service";
import authService from "../../services/auth.service";
import { extendTheme, ChakraProvider } from '@chakra-ui/react';
import {
    Table,Thead,Tbody,Tfoot,Tr,Th,Td,TableCaption,TableContainer,Grid, GridItem
} from '@chakra-ui/react'
import {Accordion, AccordionItem, AccordionButton,AccordionPanel,AccordionIcon,
} from '@chakra-ui/react'
import { Heading } from '@chakra-ui/react';
import PropTypes from 'prop-types';
import Box from '@mui/material/Box';
import Collapse from '@mui/material/Collapse';
import IconButton from '@mui/material/IconButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';

function createData(name, calories, fat, carbs, protein, price) {
    return {
        name,
        calories,
        fat,
        carbs,
        protein,
        price,
        history: [
        {
            date: '2020-01-05',
            customerId: '11091700',
            amount: 3,
        },
        {
            date: '2020-01-02',
            customerId: 'Anonymous',
            amount: 1,
        },
        ],
    };
}

    function Row(props) {
    const { row } = props;
    const [open, setOpen] = React.useState(false);

    return (
        <React.Fragment>
        <TableRow sx={{ '& > *': { borderBottom: 'unset' } }}>
            <TableCell>
            <IconButton
                aria-label="expand row"
                size="small"
                onClick={() => setOpen(!open)}
            >
                {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
            </IconButton>
            </TableCell>
            <TableCell component="th" scope="row">
            {row.name}
            </TableCell>
            <TableCell align="right">{row.calories}</TableCell>
            <TableCell align="right">{row.fat}</TableCell>
            <TableCell align="right">{row.carbs}</TableCell>
            <TableCell align="right">{row.protein}</TableCell>
        </TableRow>
        <TableRow>
            <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
            <Collapse in={open} timeout="auto" unmountOnExit>
                <Box sx={{ margin: 1 }}>
                <Typography variant="h6" gutterBottom component="div">
                    History
                </Typography>
                <Table size="small" aria-label="purchases">
                    <TableHead>
                    <TableRow>
                        <TableCell>Date</TableCell>
                        <TableCell>Customer</TableCell>
                        <TableCell align="right">Amount</TableCell>
                        <TableCell align="right">Total price ($)</TableCell>
                    </TableRow>
                    </TableHead>
                    <TableBody>
                    {row.history.map((historyRow) => (
                        <TableRow key={historyRow.date}>
                        <TableCell component="th" scope="row">
                            {historyRow.date}
                        </TableCell>
                        <TableCell>{historyRow.customerId}</TableCell>
                        <TableCell align="right">{historyRow.amount}</TableCell>
                        <TableCell align="right">
                            {Math.round(historyRow.amount * row.price * 100) / 100}
                        </TableCell>
                        </TableRow>
                    ))}
                    </TableBody>
                </Table>
                </Box>
            </Collapse>
            </TableCell>
        </TableRow>
        </React.Fragment>
    );
}

    Row.propTypes = {
    row: PropTypes.shape({
        calories: PropTypes.number.isRequired,
        carbs: PropTypes.number.isRequired,
        fat: PropTypes.number.isRequired,
        history: PropTypes.arrayOf(
        PropTypes.shape({
            amount: PropTypes.number.isRequired,
            customerId: PropTypes.string.isRequired,
            date: PropTypes.string.isRequired,
        }),
        ).isRequired,
        name: PropTypes.string.isRequired,
        price: PropTypes.number.isRequired,
        protein: PropTypes.number.isRequired,
    }).isRequired,
    };

    const rows = [
    createData('Frozen yoghurt', 159, 6.0, 24, 4.0, 3.99),
    createData('Ice cream sandwich', 237, 9.0, 37, 4.3, 4.99),
    createData('Eclair', 262, 16.0, 24, 6.0, 3.79),
    createData('Cupcake', 305, 3.7, 67, 4.3, 2.5),
    createData('Gingerbread', 356, 16.0, 49, 3.9, 1.5),
    ];

    export default function CollapsibleTable() {
    return (
        <TableContainer component={Paper}>
        <Table aria-label="collapsible table">
            <TableHead>
            <TableRow>
                <TableCell />
                <TableCell>Name</TableCell>
                <TableCell align="right">Calories</TableCell>
                <TableCell align="right">Gender</TableCell>
                <TableCell align="right">Email</TableCell>
                <TableCell align="right">Business Unit</TableCell>
                <TableCell align="right">Joined Date</TableCell>
                <TableCell align="right">Terminated Date</TableCell>
                <TableCell align="right">Location</TableCell>
                <TableCell align="right">Risk Rating</TableCell>
                <TableCell align="right">Cases</TableCell>
            </TableRow>
            </TableHead>
            <TableBody>
            {employees.map((employee) => (
                <Row key={employee.id} row={employee} />
            ))}
            </TableBody>
        </Table>
        </TableContainer>
    );
    }


const Employees = () => {
    const navigate = useNavigate();
    const [employees, setEmployees] = useState([]);
    const [tasks, setTasks] = useState([]); 
    
    const fetchEmployees = async (e) => {
        const { dataEmp } = await employeeService.viewAllEmployees();
        const employees = dataEmp;
        setEmployees(employees);
        console.log(employees);
    };

    const fetchTasksById = async(id) => {
        const { data } = await employeeService.viewAllTasksByEmployee(id);
        const tasks = data;
        setTasks(tasks);
        console.log(tasks);
    }

    useEffect(() => {
        fetchEmployees();
        try{
            employeeService.viewAllEmployees().then(
                () => {
                    console.log("ok");
                },
                (error) => {
                    console.log("Private page", error.response);
                    if (error.response && error.response.status === 403) {
                        authService.logout();
                        navigate("/auth/login");
                        window.location.reload();
                    }
                }
            );

        }catch(err){
            console.log(err);
            authService.logout();
            navigate("/auth/login");
            window.location.reload();
        }
    }, []);


    return(

        <div className="Employee">
            <div className= "div">
                <Navigation/>
            </div>
            <div className = "headingEmployee">
                <div className="title">
                    <div className="title-text">Employee Information</div>
                </div>
                <div>
                    <input type="text" className="searchbar" placeholder=" Search" />
                </div>
            </div> 

            {/* <div className="tableHeader">
                <div className="tableHeader1">Name</div>
                <div className="tableHeader2">Gender</div>
                <div className="tableHeader3">Email</div>
                <div className="tableHeader4">Business Unit</div>
                <div className="tableHeader5">Joined Date</div>
                <div className="tableHeader6">Terminated Date</div>
                <div className="tableHeader6">Location</div>
                <div className="tableHeader6">Risk Rating</div>
                <div className="tableHeader6">Cases</div>
            </div>
            <div className="straight-line"></div>
            <div className="tableEmployee">
                <ul className = "employeeList">
                    {employees.map((employee) => (
                        <li key={employee.id} className="employeeItem">
                        <div className="listRow">{employee.firstname} {employee.lastname}</div>
                        <div className="listRow">{employee.gender}</div>
                        <div className="listRow">{employee.email}</div>
                        <div className="listRow">{employee.businessUnit}</div>
                        <div className="listRow">{employee.joinedDate}</div>
                        <div className="listRow">{employee.terminatedDate}</div>
                        <div className="listRow">{employee.location}</div>
                        <div className="listRow">{employee.riskRating}</div>
                        <div className="listRow">{employee.suspectedCases}</div>
                        </li>
                    ))}
                </ul>
            

            </div> */}
            {/* // <Grid templateColumns="repeat(9, 1fr)" gap={6}>
            //     <GridItem colSpan={9}>
            //         <Heading size="md">Name</Heading>
            //         <Heading size="md">Gender</Heading>
            //         <Heading size="md">Email</Heading>
            //         <Heading size="md">Business Unit</Heading>
            //         <Heading size="md">Joined Date</Heading>
            //         <Heading size="md">Terminated Date</Heading>
            //         <Heading size="md">Location</Heading>
            //         <Heading size="md">Risk Rating</Heading>
            //         <Heading size="md">Cases</Heading>
            //     </GridItem>
            //     <Accordion allowToggle>
            //         {employees.map((employee) => ( */}
            {/* //             <AccordionItem key={employee.id}>
            //                 <AccordionButton>
            //                     <Grid templateColumns="repeat(9, 1fr)" gap={6}>
            //                         <GridItem>{employee.firstname} {employee.lastname}</GridItem>
            //                         <GridItem>{employee.gender}</GridItem>
            //                         <GridItem>{employee.email}</GridItem>
            //                         <GridItem>{employee.businessUnit}</GridItem>
            //                         <GridItem>{employee.joinedDate}</GridItem>
            //                         <GridItem>{employee.terminatedDate}</GridItem>
            //                         <GridItem>{employee.location}</GridItem>
            //                         <GridItem>{employee.riskRating}</GridItem>
            //                         <GridItem>{employee.suspectedCases}</GridItem>
            //                     </Grid>
            //                 </AccordionButton>
            //                 <AccordionPanel>
            //                     <p>hello</p>
            //                 </AccordionPanel>
            //             </AccordionItem>
            //         ))}
            //     </Accordion> */}
            {/* // </Grid> */}

            {/* <Table>
                <Thead>
                    <Th scope="col">Name</Th>
                    <Th scope="col">Gender</Th>
                    <Th scope="col">Email</Th>
                    <Th scope="col">Business Unit</Th>
                    <Th scope="col">Joined Date</Th>
                    <Th scope="col">Terminated Date</Th>
                    <Th scope="col">Location</Th>
                    <Th scope="col">Risk Rating</Th>
                    <Th scope="col">Cases</Th>
                </Thead>
                <Tbody>
                    <Accordion allowToggle>
                        {employees.map((employee) => (
                            <AccordionItem key={employee.id}>
                                <AccordionButton>
                                    <Tr>
                                        <Td>{employee.firstname} {employee.lastname}</Td>
                                        <Td>{employee.gender}</Td>
                                        <Td>{employee.email}</Td>
                                        <Td>{employee.businessUnit}</Td>
                                        <Td>{employee.joinedDate}</Td>
                                        <Td>{employee.terminatedDate}</Td>
                                        <Td>{employee.location}</Td>
                                        <Td>{employee.riskRating}</Td>
                                        <Td>{employee.suspectedCases}</Td>
                                    </Tr>
                                </AccordionButton>
                                <AccordionPanel>
                                    <p>hello</p>
                                </AccordionPanel>
                            </AccordionItem>
                        ))}
                    </Accordion>
                </Tbody>
            </Table> */}






            {/* <Table>
                <Thead>
                    <Th scope="col">Name</Th>
                    <Th scope="col">Gender</Th>
                    <Th scope="col">Email</Th>
                    <Th scope="col">Business Unit</Th>
                    <Th scope="col">Joined Date</Th>
                    <Th scope="col">Terminated Date</Th>
                    <Th scope="col">Location</Th>
                    <Th scope="col">Risk Rating</Th>
                    <Th scope="col">Cases</Th>
                </Thead>
                <Tbody>
                    <Accordion>
                        {employees.map((employee) => (
                            <Tr className="" key={employee.id}>
                                <AccordionItem>
                                    <AccordionButton>
                                        <Td>{employee.firstname} {employee.lastname}</Td>
                                        <Td>{employee.gender}</Td>
                                        <Td>{employee.email}</Td>
                                        <Td>{employee.businessUnit}</Td>
                                        <Td>{employee.joinedDate}</Td>
                                        <Td>{employee.terminatedDate}</Td>
                                        <Td>{employee.location}</Td>
                                        <Td>{employee.riskRating}</Td>
                                        <Td>{employee.suspectedCases}</Td>
                                    </AccordionButton>
                                    <AccordionPanel>
                                        <p>hello</p>
                                    </AccordionPanel>
                                </AccordionItem>
                                
                            </Tr>
                        ))}
                    </Accordion>
                </Tbody>
            </Table> */}
            {/* <table>
                <thead className ="tableHeader">
                    <tr>
                        <th >Name</th>
                        <th >Gender</th>
                        <th >Email</th>
                        <th >Business Unit</th>
                        <th >Joined Date</th>
                        <th >Terminated Date</th>
                        <th >Location</th>
                        <th >Risk Rating</th>
                        <th >Cases</th>
                    </tr>
                </thead>
                <Tbody>
                    
                    
                    <div className="straight-line"></div>
                    {employees.map((employee) => (
                    <tr className="employeeList" key={employee.id}>
                        <td>{employee.firstname} {employee.lastname}</td>
                        <td>{employee.gender}</td>
                        <td>{employee.email}</td>
                        <td>{employee.businessUnit}</td>
                        <td>{employee.joinedDate}</td>
                        <td>{employee.terminatedDate}</td>
                        <td>{employee.location}</td>
                        <td>{employee.riskRating}</td>
                        <td>{employee.suspectedCases}</td>
                    </tr>
                    ))}
                </Tbody>
            </table> */}


        </div>
        

    );
};

//export default Employees